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
function createProjectionMatrix(aspect, zNear=0.1, zFar=100.0){
  //let fieldOfView = 2*Math.atan(camera.height/2/camera.fy)*180/Math.PI;
  const fieldOfView = (45 * Math.PI) / 180; // in radians
  //const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
  //onst zNear = 0.1;
  //const zFar = 100.0;
  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
  return projectionMatrix;
}

//TODO: understand View Matrix calculation
function createModelViewMatrix() {
  const modelViewMatrix = mat4.create();
  mat4.translate(modelViewMatrix, modelViewMatrix, [-0.0, 0.0, -6.0]);
  //mat4.translate(modelViewMatrix, modelViewMatrix, camera.position);
  return modelViewMatrix;
}

function setPositionAttribute(gl, buffers, programInfo){
  const numComponents = 2;
  const type = gl.FLOAT;
  const normalize = false;
  const stride = 0;
  const offset = 0;

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
  gl.vertexAttribPointer(
    programInfo.attribLocations.vertexPosition,
    numComponents,
    type,
    normalize,
    stride,
    offset,
  );

  gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
}

function draw(gl, buffers, programInfo, canvas){
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clearDepth(1.0);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL); 
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const projectionMatrix = createProjectionMatrix(canvas.clientWidth/canvas.clientHeight);
  const modelViewMatrix = createModelViewMatrix();
  
  setPositionAttribute(gl, buffers, programInfo);
  gl.useProgram(programInfo.program);
  
  gl.uniformMatrix4fv(
    programInfo.uniformLocations.projectionMatrix,
    false,
    projectionMatrix,
  );
  gl.uniformMatrix4fv(
    programInfo.uniformLocations.modelViewMatrix,
    false,
    modelViewMatrix,
  );
  {
    const offset = 0;
    const vertexCount = 4;
    gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);
  }

}


  export { draw };