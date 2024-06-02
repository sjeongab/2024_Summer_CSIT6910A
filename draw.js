let camera = {
  width: 100,
  height: 100,
  position: [-1,-1.5,-2],
  rotation: [
      [-0.026919994628162257, -0.1565891128261527, -0.9872968974090509],
      [0.08444552208239385, 0.983768234577625, -0.1583319754069128],
      [0.9960643893290491, -0.0876350978794554, -0.013259786205163005],
  ],
  fy: 100,
  fx: 100,
};

function createProjectionMatrix(aspect, zNear=0.1, zFar=100.0){
  const fieldOfView = 2*Math.atan(camera.height/2/camera.fy)*180/Math.PI;
  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
  return projectionMatrix;
}

function createViewMatrix() {
  const viewMatrix = mat4.create();
  mat4.translate(viewMatrix, viewMatrix, camera.position);
  return viewMatrix;
}

function setPositionAttribute(gl, buffer, shader){
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.vertexBuffer);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer.indexBuffer);
  gl.vertexAttribPointer(shader.vertexPosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(shader.vertexPosition);
}

//TODO: Add colour
/*function setColorAttribute(gl, buffer, shader){
  const numComponents = 4;
  const type = gl.FLOAT;
  const normalize = false;
  const stride = 0;
  const offset = 0;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.colourBuffer);
  gl.vertexAttribPointer(
    shader.,
    numComponents,
    type,
    normalize,
    stride,
    offset
  );
  gl.enableVertexAttribArray(programInfo.attribLocations.vertexColor);

}*/

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
  const vertexCount = 36;
  const type = gl.UNSIGNED_SHORT;
  const offset = 0;
  gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
  //gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, vertexCount, vertexCount);
}


  export { draw };