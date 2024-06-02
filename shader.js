const vs = `
#version 300 es
in vec4 position;
uniform mat4 projection, view;

out vec4 vColor;
out vec4 vPosition;
void main() {
  vColor = projection * view * position;
  vPosition = position; 
  gl_Position = projection * view * position;
}
`.trim();

const fs = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec4 vPosition;

out vec4 fragColor;
void main() {
  fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
`.trim();
//TODO: Add colour
//TODO: Add texture
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function loadShaderProgram(gl) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vs);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fs);
  
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      console.error(gl.getProgramInfoLog(program));

  return program;
}

class Shader{
  constructor(gl){
    this.program = loadShaderProgram(gl);
    this.vertexPosition = gl.getAttribLocation(this.program, "position");
    //TODO: Add colour
    this.projectionMatrix = gl.getUniformLocation(this.program, "projection");
    this.viewMatrix = gl.getUniformLocation(this.program, "view");
  }
}

export {Shader};