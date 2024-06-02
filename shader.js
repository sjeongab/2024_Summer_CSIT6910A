const vs = `
#version 300 es
in vec4 position;
in vec4 colour;
uniform mat4 projection, view;

out vec4 vColour;
out vec4 vPosition;
void main() {
  vColour = colour;
  vPosition = position; 
  gl_Position = projection * view * position;
}
`.trim();

const fs = `
#version 300 es
precision highp float;

in vec4 vColour;
in vec4 vPosition;

out vec4 fragColor;
void main() {
  fragColor = vColour;
}
`.trim();

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
    this.vertexColour = gl.getAttribLocation(this.program, "colour");
    this.projectionMatrix = gl.getUniformLocation(this.program, "projection");
    this.viewMatrix = gl.getUniformLocation(this.program, "view");
  }
}

export {Shader};