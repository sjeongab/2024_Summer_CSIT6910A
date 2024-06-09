import {Shader} from "./shader.js";
import {Buffer} from "./buffer.js";
import {draw} from "./draw.js";
import {parseOBJ} from "./loadObj.js";

function render(gl, buffer, shader, then){
  function rotate(now) {
    now*=0.001;
    deltaTime = now - then;
    then = now;
    draw(gl, buffer, shader, canvas, cubeRotation);
    cubeRotation += deltaTime*2;
    requestAnimationFrame(rotate);
  }
  requestAnimationFrame(rotate);
}

let deltaTime = 0;
let cubeRotation = 0.0;
async function main() {
  //const box = await (await fetch('./box.obj')).text();
  //const ham0 = await (await fetch('./ham_0.obj')).text();
  //const ham1 = await (await fetch('./ham_1.obj')).text();
  //const ham2 = await (await fetch('./ham_2.obj')).text();
  const objHref = 'https://webglfundamentals.org/webgl/resources/models/windmill/windmill.obj';  
  const response = await fetch(objHref);
  const text = await response.text();
  const obj = parseOBJ(text);
  //console.log(box);
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