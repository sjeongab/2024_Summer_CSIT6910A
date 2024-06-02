import {Shader} from "./shader.js";
import {Buffer} from "./buffer.js";
import {draw} from "./draw.js";



async function main() {
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl2", {antialias: false,});
  const shader = new Shader(gl);
  const buffer = new Buffer(gl);

  draw(gl, buffer, shader, canvas);
}

main().catch((err) => {
    console.log(err);
});