"""ct2skel: fit the SKEL body/skeleton model to a preoperative CT and export STL parts.

Pipeline
--------
DICOM series ─▶ HU volume ─▶ skin / bone masks (+ optional TotalSegmentator labels)
             ─▶ surface meshes ─▶ SKEL fit (pose, shape, translation)
             ─▶ STL export (CT skin, CT bones, SKEL skin, SKEL bones per part)
             ─▶ manifest.json + self-contained web viewer (overlay / side-by-side / exploded)
"""

__version__ = "0.1.0"
