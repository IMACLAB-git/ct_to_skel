"""Volume loading (DICOM series / NIfTI / NRRD / MHA) and DICOM writing.

All volumes are represented by :class:`Volume` with the array in (z, y, x) order
(SimpleITK convention) and Hounsfield units.  World coordinates are DICOM LPS
millimetres.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class Volume:
    array: np.ndarray            # (Z, Y, X) float32, Hounsfield units
    spacing: tuple               # (sx, sy, sz) mm
    origin: tuple                # (ox, oy, oz) mm, LPS
    direction: np.ndarray        # 3x3 direction cosines (row-major, sitk order)
    meta: dict | None = None

    # ------------------------------------------------------------------ geometry
    @property
    def shape_xyz(self) -> tuple:
        z, y, x = self.array.shape
        return (x, y, z)

    @property
    def affine(self) -> np.ndarray:
        """4x4 matrix mapping continuous index (i, j, k) = (x, y, z) to LPS mm."""
        A = np.eye(4)
        A[:3, :3] = self.direction @ np.diag(self.spacing)
        A[:3, 3] = self.origin
        return A

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """ijk: (N, 3) continuous indices in (x, y, z) order -> (N, 3) LPS mm."""
        ijk = np.asarray(ijk, dtype=np.float64)
        return ijk @ (self.direction @ np.diag(self.spacing)).T + np.asarray(self.origin)

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64) - np.asarray(self.origin)
        M = np.linalg.inv(self.direction @ np.diag(self.spacing))
        return xyz @ M.T

    def zyx_to_world(self, zyx: np.ndarray) -> np.ndarray:
        """Array-order indices (z, y, x) -> LPS mm."""
        zyx = np.asarray(zyx, dtype=np.float64)
        return self.index_to_world(zyx[:, ::-1])

    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    # ------------------------------------------------------------------ sitk
    def to_sitk(self, array: np.ndarray | None = None) -> sitk.Image:
        arr = self.array if array is None else array
        img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
        img.SetSpacing(tuple(float(s) for s in self.spacing))
        img.SetOrigin(tuple(float(o) for o in self.origin))
        img.SetDirection(tuple(float(d) for d in self.direction.flatten()))
        return img

    @classmethod
    def from_sitk(cls, img: sitk.Image, meta: dict | None = None) -> "Volume":
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return cls(
            array=arr,
            spacing=tuple(img.GetSpacing()),
            origin=tuple(img.GetOrigin()),
            direction=np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3),
            meta=meta or {},
        )


# ---------------------------------------------------------------------- loading
_VOLUME_EXT = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")


def _is_volume_file(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(e) for e in _VOLUME_EXT)


def load_volume(path: str | os.PathLike) -> Volume:
    """Load a CT volume from a DICOM directory or a single volume file.

    For a directory containing several DICOM series, the series with the most
    slices is used.
    """
    p = Path(path)
    if p.is_file():
        if not _is_volume_file(p):
            # single DICOM file -> load the series it belongs to
            return _load_dicom_dir(p.parent)
        try:
            img = sitk.ReadImage(str(p))
        except RuntimeError as e:
            if "orthonormal" not in str(e):
                raise
            return _load_nifti_nibabel(p)      # sheared / non-orthonormal sform: ITK refuses, nibabel copes
        return Volume.from_sitk(img, meta={"source": str(p)})
    if p.is_dir():
        return _load_dicom_dir(p)
    raise FileNotFoundError(path)


def _load_nifti_nibabel(p: Path) -> Volume:
    """Load a NIfTI whose affine is not orthonormal (ITK rejects it) by orthonormalising the rotation part.

    The voxel array is kept; the direction is replaced by the closest rotation (polar
    decomposition), which is exact for pure flips/permutations and a mm-level approximation
    for slightly sheared headers.
    """
    import nibabel as nib

    img = nib.load(str(p))
    data = np.asanyarray(img.dataobj).astype(np.float32)            # (x, y, z)
    aff = img.affine.astype(np.float64)                              # index -> RAS mm
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    A = ras_to_lps @ aff                                             # index -> LPS mm
    M = A[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    R = M / spacing
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return Volume(array=np.ascontiguousarray(data.transpose(2, 1, 0)), spacing=tuple(float(v) for v in spacing),
                  origin=tuple(float(v) for v in A[:3, 3]), direction=R_ortho,
                  meta={"source": str(p), "loader": "nibabel(orthonormalised)"})


def _load_dicom_dir(d: Path) -> Volume:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(d))
    if not series_ids:
        # maybe a nested directory layout: search recursively
        candidates = []
        for sub in d.rglob("*"):
            if sub.is_dir():
                ids = reader.GetGDCMSeriesIDs(str(sub))
                for sid in ids:
                    candidates.append((sub, sid))
        if not candidates:
            raise RuntimeError(f"No DICOM series found under {d}")
        best = None
        for sub, sid in candidates:
            files = reader.GetGDCMSeriesFileNames(str(sub), sid)
            if best is None or len(files) > len(best[2]):
                best = (sub, sid, files)
        d, sid, files = best
    else:
        best = None
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if best is None or len(files) > len(best[1]):
                best = (sid, files)
        sid, files = best

    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    img = reader.Execute()
    if img.GetNumberOfComponentsPerPixel() != 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    img = sitk.Cast(img, sitk.sitkFloat32)   # rescale slope/intercept already applied by GDCM

    meta = {"source": str(d), "series_uid": sid, "n_slices": len(files)}
    for key, tag in (("patient_sex", "0010|0040"), ("patient_age", "0010|1010"),
                     ("modality", "0008|0060"), ("study_desc", "0008|1030"),
                     ("patient_position", "0018|5100")):
        try:
            meta[key] = reader.GetMetaData(0, tag).strip()
        except Exception:
            pass
    return Volume.from_sitk(img, meta=meta)


def save_volume(vol: Volume, path: str | os.PathLike, array: np.ndarray | None = None) -> None:
    """Save the volume (or an alternative array on the same grid) to NIfTI/NRRD/MHA."""
    sitk.WriteImage(vol.to_sitk(array), str(path))


# ---------------------------------------------------------------------- dicom writing
def write_dicom_series(vol: Volume, out_dir: str | os.PathLike, patient_name: str = "PHANTOM^CT",
                       patient_sex: str = "M") -> list[str]:
    """Write a Volume as a CT DICOM series (one file per axial slice).

    Intended for producing test data; HU are stored as int16 with rescale
    intercept 0 / slope 1.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    now = datetime.datetime.now()
    nz = vol.array.shape[0]
    files = []
    row_dir = vol.direction[:, 0]
    col_dir = vol.direction[:, 1]
    for k in range(nz):
        sl = np.clip(np.round(vol.array[k]), -32768, 32767).astype(np.int16)
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.Modality = "CT"
        ds.PatientName, ds.PatientID, ds.PatientSex = patient_name, "PHANTOM0001", patient_sex
        ds.PatientPosition = "HFS"
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.ImagePositionPatient = [float(v) for v in vol.index_to_world(np.array([[0, 0, k]]))[0]]
        ds.ImageOrientationPatient = [float(v) for v in np.concatenate([row_dir, col_dir])]
        ds.PixelSpacing = [float(vol.spacing[1]), float(vol.spacing[0])]   # row spacing (y), col spacing (x)
        ds.SliceThickness = float(vol.spacing[2])
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.Rows, ds.Columns = sl.shape
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0.0, 1.0
        ds.WindowCenter, ds.WindowWidth = 40.0, 400.0
        ds.PixelData = sl.tobytes()
        fn = out / f"IM{k + 1:04d}.dcm"
        ds.save_as(str(fn), write_like_original=False)
        files.append(str(fn))
    return files
