Changes for this version of DIPLOMAT:
- reworked occlusion logic to support decay, merge current visible and prior occluded probabilities through a maximum.
- enter state is now being used again; patched a bug that made DIPLOMAT unable to use enter state when frames are missing body parts.
- patched domination logic to avoid NaN probabilities when complete domination occurs immediately after the enter state is used.
- applied sidebar radiobutton color fix from interactive mode to tweak mode. UI colors now update when the tracking colors update.
