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

function loadShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function initShaderProgram(gl) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vs);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fs);
  
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      console.error(gl.getProgramInfoLog(program));

  return program
}

function getShaderProgramInfo(gl, shaderProgram){
  const programInfo = {
    program: shaderProgram,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(shaderProgram, "position"),
    },
    uniformLocations:{
      projectionMatrix: gl.getUniformLocation(shaderProgram, "projection"),
      modelViewMatrix: gl.getUniformLocation(shaderProgram, "view"),
    },
  };
  return programInfo;
};

export {initShaderProgram, getShaderProgramInfo};