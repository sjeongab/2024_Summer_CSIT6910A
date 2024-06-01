function initBuffer(gl) {
    const positionBuffer = initPositionBuffer(gl);
  
    return {
      position: positionBuffer,
    };
  }

function initPositionBuffer(gl) {
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);

    const positions = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    return indexBuffer;
}

export {initBuffer};