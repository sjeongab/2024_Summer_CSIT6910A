import {Shader} from "./shader.js";
import {Buffer} from "./buffer.js";
import {draw} from "./draw.js";



let deltaTime = 0;
let cubeRotation = 0.0;
async function main() {
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl2", {antialias: false,});
  const shader = new Shader(gl);
  const buffer = new Buffer(gl);
  let then = 0;
  function render(now) {
    now*=0.001;
    deltaTime = now - then;
    then = now;
    draw(gl, buffer, shader, canvas, cubeRotation);
    cubeRotation += deltaTime;
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
  
}

main().catch((err) => {
    console.log(err);
});