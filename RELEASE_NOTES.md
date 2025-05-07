DIPLOMAT version 0.3.0 represents a major refactoring of the entire frontend 
system, cli interface, and both the SLEAP and DLC frontends. The following 
adjustments have been made:
 - Frontends reduced to only needing to implement 2 functions to support all inference functionality in DIPLOMAT.
 - Almost all DIPLOMAT cli commands and core functions are now frontend independent, and all use diplomat's csv format instead of frontend specific formats.
 - The SLEAP frontend does not require sleap to be installed, and uses onnx for inference.
 - The DLC frontend does not require deeplabcut to be installed, and uses onnx for inference.
 - Several improvements made to DIPLOMAT's docs, including a simplified installation process.