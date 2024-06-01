import {Shader} from "./shader.js";
import {loadBuffer} from "./buffer.js";
import {draw} from "./draw.js";


async function main() {
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl2", {antialias: false,});
  const shader = new Shader(gl);
  const positions = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
  const buffer = loadBuffer(gl, positions);

  draw(gl, buffer, shader, canvas);
}

main().catch((err) => {
    console.log(err);
});