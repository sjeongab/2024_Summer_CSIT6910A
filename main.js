import {vs, fs, initShaderProgram} from "./shader.js";
import {initBuffer} from "./buffer.js";
import {createProjectionMatrix} from "./draw.js";


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


async function main() {
  const canvas = document.getElementById("canvas");
  const shaderProgram = initShaderProgram(gl, vs, fs);

  background();
  createProjectionMatrix();
}

main().catch((err) => {
    console.log(err);
});