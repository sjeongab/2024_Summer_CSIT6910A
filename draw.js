let camera = {
  width: 100,
  height: 100,
  position: [5, 0, 1],
  rotation: [
      [-0.026919994628162257, -0.1565891128261527, -0.9872968974090509],
      [0.08444552208239385, 0.983768234577625, -0.1583319754069128],
      [0.9960643893290491, -0.0876350978794554, -0.013259786205163005],
  ],
  fy: 100,
  fx: 100,
};

//TODO: understand Projection Matrix calculation
function createProjectionMatrix(zNear=0.1, zFar=100.0){
  let fieldOfView = 2*Math.atan(camera.height/2/camera.fy)*180/Math.PI;
  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
  return projectionMatrix;
}

//TODO: understand View Matrix calculation
function createModelViewMatrix(camera) {
  const modelViewMatrix = mat4.create();
}

  export { createProjectionMatrix };