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
function createViewMatrix() {
  const viewMatrix = mat4.create();
  mat4.translate(viewMatrix, viewMatrix, [-0.0, 0.0, -6.0]);
  //mat4.translate(modelViewMatrix, modelViewMatrix, camera.position);
  return viewMatrix;
}

function setPositionAttribute(gl, buffer, shader){
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.vertexAttribPointer(shader.vertexPosition, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(shader.vertexPosition);
}

function draw(gl, buffer, shader, canvas){
  gl.disable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const projectionMatrix = createProjectionMatrix(canvas.clientWidth/canvas.clientHeight);
  const viewMatrix = createViewMatrix();
  
  setPositionAttribute(gl, buffer, shader);
  gl.useProgram(shader.program);
  
  gl.uniformMatrix4fv(shader.projectionMatrix, false, projectionMatrix,);
  gl.uniformMatrix4fv(shader.viewMatrix,false,viewMatrix,);
  //TODO: update vertexCount
  const vertexCount = 4;
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, vertexCount);
}


  export { draw };