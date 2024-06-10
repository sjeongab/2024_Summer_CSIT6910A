let camera = {
  width: 100,
  height: 100,
  position: [0,1,0],
  rotation: [
    [0,-0.1,0],
    [0,0.1,0],
    [0,0,0],
      /*[-0.026919994628162257, -0.1565891128261527, -0.9872968974090509],
      [0.08444552208239385, 0.983768234577625, -0.1583319754069128],
      [0.9960643893290491, -0.0876350978794554, -0.013259786205163005],*/
  ],
  fy: 100,
  fx: 100,
};

document.addEventListener('keydown', onKeyDown, false);
let angle = 0.0;
let radius=200;
let y = 0;
function onKeyDown(event){
  if(event.key == 'a'){
    radius += 1;
    camera.position[0] += 2;
  }
  if(event.key == 'w'){
    y+=1;
    camera.position[1] += 2;
  }
  if(event.key == 's'){
    y-=1;
    camera.position[1] -= 2;
  }
  if(event.key == 'd'){
    camera.position[0] -= 2;
  }
  if(event.key == 'q'){
    camera.position[2] += 2;
  }
  if(event.key == 'e'){
    camera.position[2] -= 2;
  }
}


function createProjectionMatrix(gl, zNear=0.1, zFar=100.0){
  var aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
  var fieldOfViewRadians = degToRad(60);
  return m4.perspective(fieldOfViewRadians, aspect, zNear, zFar);
  return [
      [(2 * camera.fx) / camera.width, 0, 0, 0],
      [0, -(2 * camera.fy) / camera.height, 0, 0],
      [0, 0, zFar / (zFar - zNear), 1],
      [0, 0, -(zFar * zNear) / (zFar - zNear), 0],
  ].flat();
  
}

function degToRad(d) {
  return d * Math.PI / 180;
}

function createViewMatrix(camera) {
  const R = camera.rotation.flat();
  //console.log(R);
  const t = camera.position;
  const camToWorld = [
      [R[0], R[1], R[2], 0],
      [R[3], R[4], R[5], 0],
      [R[6], R[7], R[8], 0],
      [
          -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
          -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
          -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
          1,
      ],
  ].flat();
  
  var cameraAngleRadians = degToRad(0);
  var cameraMatrix = m4.yRotation(cameraAngleRadians);
  cameraMatrix = m4.translate(cameraMatrix, 0, 0, radius * 1.5);
  return cameraMatrix;
  return camToWorld;
}

function setPositionAttribute(gl, buffer, shader){
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.vertexBuffer);
  //gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer.indexBuffer);
  gl.vertexAttribPointer(shader.vertexPosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(shader.vertexPosition);
}

function setColourAttribute(gl, buffer, shader){
  const numComponents = 4;
  const type = gl.FLOAT;
  const normalize = false;
  const stride = 0;
  const offset = 0;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.colourBuffer);
  gl.vertexAttribPointer(
    shader.vertexColour,
    numComponents,
    type,
    normalize,
    stride,
    offset
  );
  gl.enableVertexAttribArray(shader.vertexColour);

}

function draw(gl, buffer, shader, cubeRotation){
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const projectionMatrix = createProjectionMatrix(gl);
  const viewMatrix = createViewMatrix(camera);
  mat4.rotate(
    viewMatrix, // destination matrix
    viewMatrix, // matrix to rotate
    cubeRotation, // amount to rotate in radiansssss
    [0, 1, 0]
  );
  
  setPositionAttribute(gl, buffer, shader);
  //setColourAttribute(gl, buffer, shader);
  gl.useProgram(shader.program);
  
  gl.uniformMatrix4fv(shader.projectionMatrix, false, projectionMatrix,);
  gl.uniformMatrix4fv(shader.viewMatrix,false,viewMatrix,);
  //TODO: update vertexCount
  const vertexCount = 7038;
  const type = gl.UNSIGNED_SHORT;
  const offset = 0;
  //gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
  gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, vertexCount, vertexCount);
}


  export { draw };