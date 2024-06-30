import {Shader} from "./shader.js";
import {Buffer} from "./buffer.js";
import {draw} from "./draw.js";
import {parseOBJ, Figure} from "./loadObj.js";

const fpsElem = document.querySelector("#fps");


function render(gl, buffer, shader, then){
  function rotate(now) {
    now *= 0.001; 
    const deltaTime = now - then;          // compute time since last frame
    then = now;                            // remember time for next frame
    const fps = 1 / deltaTime;             // compute frames per second
    fpsElem.textContent = fps.toFixed(1);
    draw(gl, buffer, shader, now);
    requestAnimationFrame(rotate);
  }
  requestAnimationFrame(rotate);
}

let deltaTime = 0;
let cubeRotation = 0.0;
async function main() {
  const objHref = 'https://webgl2fundamentals.org/webgl/resources/models/windmill/windmill.obj';  
  const response = await fetch(objHref);
  const text = await response.text();
  const figure = new Figure(text);
  console.log(figure.geometries);
  //const obj = parseOBJ(text);
  //console.log(obj);
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl2", {antialias: false,});
  const shader = new Shader(gl);
  const buffer = new Buffer(gl, obj.position);
  let then = 0;
  render(gl, buffer, shader, then);
  
}

main().catch((err) => {
    console.log(err);
});