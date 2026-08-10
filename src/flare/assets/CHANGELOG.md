# Changelog – AgileX PiPER Description

All notable changes to this model will be documented in this file.

## [2026-05-27]
- Added `ee_tip` site inside `link6` body at `pos="0 0 0.12"` (link6 Z-axis, gripper pad midpoint).
  Used as default EE reference for `policy_verify_3d.py` (`--ee-site ee_tip`).
  Adjust Z value (0.10–0.135 range) to match actual gripper geometry if needed.

## [2025-02-16]
- Initial release.
