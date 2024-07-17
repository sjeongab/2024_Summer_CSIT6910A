# 2024 Summer CSIT 6910A
**Re-implementation of Seathru NeRF through 3D Gaussian Splatting**

## WarmUp
### TODO
* Code clean up
* Optimize code: render faster
* Revisit projectionMatrix / viewMatrix
* Add texture to object
* Add controls to the viewer
* Create CSS file
* pass geometry object
* Add how to
* Add requirements

## 3D Gaussian Splatting SeathruNeRF
### TODO
* Code clean up
* Add parameter for depth map and medium
* Implement view-input MLP
* Add cam direction as input

## Idea
* Group watercolour by splatting group(at rasterization) to increase rendering speed?

## LOG
**[2024-05-22]** 
  * Add shader and colour background with buffer
    
**[2024-05-27]**
  * Create projection matrix based on camera
    
**[2024-06-01]**
  * Fix shader and add draw object to screen
    
**[2024-06-02]**
  * Add camera
  * Render 3D object
  * Add colour to object
  * Animate object

**[2024-06-05]**
  * Proposal: Background & Technical Method

**[2024-06-06]**
  * Proposal: Expected Result & Timeline

**[2024-06-08]**
  * Install WSL for seathruNeRF

**[2024-06-09]**
  * Load windmill obj

**[2024-06-10]**
  * Add keyboard input
  * Start re-work on loadObj

**[2024-06-12]**
  * SeathruNeRF env set up: Install JaxLib and Cuda

**[2024-06-13]**
  * SeathruNeRF env set up: Tensorflow 

**[2024-06-16]**
  * 3DGS literature review: introduction

**[2024-06-17]**
  * WarmUp: parse keyword

**[2024-06-18]**
  * Try NeRF_studio seathru_NeRF

**[2024-06-19]**
  * Re-install Linux

**[2024-06-20]**
  * Successfully installed NeRF_studio seathru_NeRF

**[2024-06-21]**
  * 3DGS literature review: Related work

**[2024-06-22]**
  * 3DGS literature review: Optimization
  * WarmUp: Load OBJ

**[2024-06-23]**
  * Training SeathruNeRF via nerfstudio

**[2024-06-28]**
  * Training SeathruNeRF via nerfstudio

**[2024-06-29]**
  * Render SeathruNeRF

**[2024-06-30]**
  * 3DGS: code re-implementation start
  
**[2024-07-01]**
  * Install 3DGS

**[2024-07-02]**
  * Install and run 3DGS

**[2024-07-03]**
  * Render fixed Seathru 3DGS

**[2024-07-04]**
  * Render fixed seathru 3DGS WebGL

**[2024-07-04]**
  * 3DGS with learnable medium variable

**[2024-07-10]**
  * Install tinycudnn

**[2024-07-11]**
  * 3DGS with learnable medium variable

**[2024-07-12]**
  * 3DGS with learnable medium variable

**[2024-07-15]**
  * MLP: Analysis on cam direction