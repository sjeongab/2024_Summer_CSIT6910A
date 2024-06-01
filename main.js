import {initShaderProgram, getShaderProgramInfo} from "./shader.js";
import {initBuffer} from "./buffer.js";
import {draw} from "./draw.js";


const gl = canvas.getContext("webgl2", {
  antialias: false,
});

function background(){
  let blue = 0.0;
  let then = 0;
  let deltaTime;
  function render(now) {
    now *= 0.001; // convert to seconds
    deltaTime = now - then;
    then = now;
    blue += deltaTime;
    gl.clearColor(0.0, 0.0, blue, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

/*const programInfo = {
  program: shaderProgram,
  attribLocations: {
    vertexPosition: gl.getAttribLocation(shaderProgram, "aVertexPosition"),
  },
  uniformLocations: {
    projectionMatrix: gl.getUniformLocation(shaderProgram, "uProjectionMatrix"),
    modelViewMatrix: gl.getUniformLocation(shaderProgram, "uModelViewMatrix"),
  },
};*/

async function main() {
  const canvas = document.getElementById("canvas");
  const shaderProgram = initShaderProgram(gl);
  const shaderProgramInfo = getShaderProgramInfo(gl, shaderProgram);
  const buffer = initBuffer(gl);

  //background();
  draw(gl, buffer, shaderProgramInfo, canvas);
}

main().catch((err) => {
    console.log(err);
});