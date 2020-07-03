"use strict";

function isInViewport(element) {
  var rect = element.getBoundingClientRect();
  var html = document.documentElement;
  var w = window.innerWidth || html.clientWidth;
  var h = window.innerHeight || html.clientHeight;
  return rect.top < h && rect.left < w && rect.bottom > 0 && rect.right > 0;
}

  // Adds the WASM backend to the global backend registry.
  //import '@tensorflow/tfjs-backend-wasm';
  // Set the backend to WASM and wait for the module to be ready.
export function mnistDemo(divId, canvasId) {
  const root = document.getElementById(divId);
  const $ = q => root.querySelector(q);
  const $$ = q => root.querySelectorAll(q);
  const mnistCanvas = document.createElement('canvas');
  const mnistCtx = mnistCanvas.getContext('2d');

  // side length in CA space
  const D = 28 * 2;
  const numChannel = 20;
  const state = tf.variable(tf.zeros([1, D, D, numChannel]));
  let currDig = 0;
  let currSample = 0;
  let uiDigitSamples = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let eraser = false;
  let paused = false;
  let visibleChannel = -1;
  let firingChance = 0.5;
  let drawRadius = 2.5;
  const colors = [
      [128, 0, 0],
      [230, 25, 75],
      [70, 240, 240],
      [210, 245, 60],
      [250, 190, 190],
      [170, 110, 40],
      [170, 255, 195],
      [165, 163, 159],
      [0, 128, 128],
      [128, 128, 0],
      [0, 0, 0], // This is the default for digits.
      [255, 255, 255] // This is the background.
  ];
  const colorLookup = tf.tensor(colors);

  async function loadMnistSamples() {

    async function toBmp(url) {
        return new Promise((resolve,reject) => {
            let img = document.createElement('img');
            img.addEventListener('load', function() {
                resolve(this);
            });
            img.src = url;
        });
    }
    const mnistBmp = await toBmp("mnist.png");  //await (await fetch("mnist.png")).blob());
    mnistCanvas.width = mnistBmp.width;
    mnistCanvas.height = mnistBmp.height;
    mnistCtx.drawImage(mnistBmp,0,0);  
  }
  
  function rgb(values) {
    return 'rgb(' + values.join(', ') + ')';
  }

  function getDigit(digit, sample) {
    const x = sample * 28;
    const y = digit * 28;
    return mnistCtx.getImageData(x, y, 28, 28);
  };

  function getDigitTF(digit, sample) {
    return tf.tensor(new Float32Array(getDigit(digit, sample).data)).reshape([1,28,28,4]).slice([0,0,0,2], [1,28,28,1]).div(255.0);
  };

  function reset() {
      const digit = getDigitTF(currDig, currSample);
      const padding = (D - 28)/2.0;
      state.assign(tf.pad(digit, [[0,0], [padding, padding], [padding, padding], [0, 19]]));
  }

  function switcheroo() {
      const digit = getDigitTF(currDig, currSample);
      const padding = (D - 28)/2.0;
      const paddedDigit = tf.pad(digit, [[0,0], [padding, padding], [padding, padding], [0,0]])
      const mask = paddedDigit.greater(0.1).asType('float32');
      const maskedOldState = state.slice([0,0,0,1], [1, D, D, 19]).mul(mask);
      state.assign(tf.concat([paddedDigit, maskedOldState], 3));
  }

  //hacky way to uncolor last thing.
  async function initUI() {
    await loadMnistSamples();
    for (let i = 0; i < 10; i++) {
      const dcv = document.createElement('canvas');
      dcv.id = "digit-" + i;
      dcv.width = 28;
      dcv.height = 28;
      const dctx = dcv.getContext('2d');
      if (i != 0) {
        dctx.putImageData(getDigit(i, 0), 0, 0)
      } else {
        dctx.putImageData(getDigit(i, 1), 0, 0)
        uiDigitSamples[i] += 1;
      }
      dctx.globalCompositeOperation='difference';
      dctx.fillStyle = 'white';
      dctx.fillRect(0, 0, 28, 28);
      dctx.globalCompositeOperation = "screen";
      dctx.fillStyle = rgb(colors[i]);
      dctx.fillRect(0, 0, 28, 28);
      dcv.onclick = () => {
        // update the digit to show
        currDig = i;
        currSample = uiDigitSamples[i];
        switcheroo();
        // paint our legend with next digit
        uiDigitSamples[i] = (uiDigitSamples[i] + 1) % 20;
        dctx.putImageData(getDigit(i, uiDigitSamples[i]), 0, 0)
        dctx.globalCompositeOperation='difference';
        dctx.fillStyle = 'white';
        dctx.fillRect(0, 0, 28, 28);
        dctx.globalCompositeOperation = "screen";
        dctx.fillStyle = rgb(colors[i]);
        dctx.fillRect(0, 0, 28, 28);
        console.log("clicked" + i);
      }
      $('#pattern-selector').appendChild(dcv);
    }
    console.log("loaded");
    $('#reset').onclick = reset;
    $('#play-pause').onclick = () => {
      paused = !paused;
      updateUI();
    };
    $('#eraser-pencil').onclick = () => {
      eraser = !eraser;
      updateUI();
    };
    console.log("loaded");
    $('#speed').onchange = updateUI;
    $('#speed').oninput = updateUI;
    updateUI();
  };
  
  function updateUI() {
    // $$('#model-hints span').forEach(e => {
    //   e.style.display = e.id.startsWith(experiment) ? "inline" : "none";
    // });
    $('#play').style.display = paused ? "inline" : "none";
    $('#pause').style.display = !paused ? "inline" : "none";
    $('#eraser').style.display = !eraser ? "inline-block" : "none";
    $('#pencil').style.display = eraser ? "inline-block" : "none";
    const speed = parseInt($('#speed').value);
    $('#speedLabel').innerHTML = ['1/60 x', '1/10 x', '1/2 x', '1x', '2x', '4x', '<b>max</b>'][speed + 3];
  };

  const parseConsts = model_graph => {
    const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
                    'DT_FLOAT':['float32', 'floatVal', Float32Array]};
    
    const consts = {};
    model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
      const v = node.attr.value.tensor;
      const [dtype, field, arrayType] = dtypes[v.dtype];
      if (!v.tensorShape.dim) {
        consts[node.name] = [tf.scalar(v[field][0], dtype)];
      } else {
        const shape = v.tensorShape.dim.map(d=>parseInt(d.size));
        let arr;
        if (v.tensorContent) {
          const data = atob(v.tensorContent);
          const buf = new Uint8Array(data.length);
          for (var i=0; i<data.length; ++i) {
            buf[i] = data.charCodeAt(i);
          }
          arr = new arrayType(buf.buffer);
        } else {
          const size = shape.reduce((a, b)=>a*b);
          arr = new arrayType(size);
          arr.fill(v[field][0]);
        }
        consts[node.name] = [tf.tensor(arr, shape, dtype)];
      }
    }));
    return consts;
  }
  

  $('#brushSlider').oninput = (e) => {
      drawRadius = parseFloat(e.target.value)/2.0;
      $('#radius').innerText =drawRadius;
  };

  let backgroundWhite = true;

  const run = async () => {
      await initUI();
      const r = await fetch("model.json");
      const consts = parseConsts(await r.json());
      const model = await tf.loadGraphModel("model.json");
      Object.assign(model.weights, consts);
      const digZero = getDigitTF(0, 0);
      const padding = (D - 28)/2.0;
      state.assign(tf.pad(digZero, [[0,0], [padding, padding], [padding, padding], [0, 19]]))
      
      const [_, h, w, ch] = state.shape;
      console.log(state.shape);

      const scale = 10;

      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      canvas.width = w * scale;
      canvas.height = h * scale;

      const drawing_canvas = document.createElement('canvas');
      drawing_canvas.width = w;
      drawing_canvas.height = h;
      const render_canvas = document.createElement('canvas');
      render_canvas.width = w;
      render_canvas.height = h;
      const draw_ctx = drawing_canvas.getContext('2d');
      const render_ctx = render_canvas.getContext('2d');

      // Useful for understanding background color.
      
      //let blackAndWhite = tf.zeros();//.fill(0.01);
      let arr = new Float32Array(h * w * 2);
      arr.fill(0.01);
      const blackAndWhiteFull = tf.tensor(arr, [1,h,w,2], tf.float32)

      const drawCanvas = (imgd, e) => {
          var matrix = [];
          for(let i=0; i<imgd.width; i++) {
              matrix[i] = [];
              for(let j=0; j<imgd.height; j++) {
                  let intensity = imgd.data[(imgd.height*j*4 + i*4)];
                  // For drawing, we want to add shades of grey. For erasing, we don't.
                  if (!e.shiftKey || eraser) {
                    intensity *= (imgd.data[(imgd.height*j*4 + i*4 + 3)] / 255);
                  }
                  matrix[i][j] = intensity;
              }
          }

          tf.tidy(() => {
              const stroke = tf.tensor(matrix).transpose().toFloat().div(255.).expandDims(0).expandDims(3);
              const stroke_pad = tf.concat([stroke, tf.zeros([1, h, w, ch-1])], 3);
              const mask = tf.tensor(1.).sub(stroke);
              if (e.shiftKey || eraser) {
                  state.assign(state.mul(mask));
              } else {
                  state.assign(state.mul(mask).add(stroke_pad));
              }
          });

          // Then clear the canvas.
          draw_ctx.clearRect(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
      }

      const line = (x0, y0, x1, y1, r, e) => {
          draw_ctx.beginPath();
          draw_ctx.moveTo(x0, y0);
          draw_ctx.lineTo(x1, y1);
          draw_ctx.strokeStyle = "#ff0000";
          // Erasing has a much larger radius.
          draw_ctx.lineWidth = ((e.shiftKey || eraser)? 5. * r : r);
          draw_ctx.stroke();

          const imgd = draw_ctx.getImageData(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
          drawCanvas(imgd, e);
      }


      const circle = (x, y, r, e) => {
          draw_ctx.beginPath();

          const drawRadius = ((e.shiftKey || eraser) ? 5. * r : r) / 3.;

          draw_ctx.arc(x, y, drawRadius, 0, 2 * Math.PI, false);
          draw_ctx.fillStyle = "#ff0000";
          draw_ctx.fill();
          draw_ctx.lineWidth = 1;
          draw_ctx.strokeStyle = "#ff0000";
          draw_ctx.stroke();

          const imgd = draw_ctx.getImageData(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
          drawCanvas(imgd, e);
      }

      const draw_r = 2.0;


      function canvasToGrid(xin, yin) {
        const x = Math.floor(((xin / canvas.clientWidth) * canvas.width) / scale); //(e.pageX-e.target.offsetLeft) / scale);
        const y = Math.floor(((yin / canvas.clientHeight) * canvas.height) / scale); //(e.pageY-e.target.offsetTop) / scale);
        return [x, y];    
      }

      const getClickPos = e=>{
          return canvasToGrid(e.offsetX, e.offsetY);
      }

      function getTouchPos(touch) {
        const rect = canvas.getBoundingClientRect();
        return canvasToGrid(touch.clientX - rect.left, touch.clientY - rect.top);
      }

      let lastX = 0;
      let lastY = 0;

      canvas.onmousedown = e => {
          const [x, y] = getClickPos(e);
          lastX = x;
          lastY = y;
          circle(x,y,drawRadius, e);
      }
      canvas.onmousemove = e => {
          const [x, y] = getClickPos(e);
          if (e.buttons == 1) {
              line(lastX,lastY, x,y,drawRadius, e);
          }
          lastX = x;
          lastY = y;
      }

      canvas.addEventListener("touchstart", e => {
        e.preventDefault();
        const [x, y] = getTouchPos(e.changedTouches[0]);
        lastX = x;
        lastY = y;
        circle(x,y,drawRadius, e);
      });

      canvas.addEventListener("touchmove", e => {
        e.preventDefault();
          for (const t of e.touches) {
            const [x, y] = getTouchPos(t);
            console.log([x,y]);
            line(lastX,lastY, x,y,drawRadius, e);
            lastX = x;
            lastY = y;
          }
        });

    
      let lastDrawTime = 0;
      let lastStepTime = 0;
      let lastStepCount = 0;
      let stepsPerFrame = 1;
      let frameCount = 0;

      let first = true;

      const render = async (time) => {
        if (!paused && isInViewport(canvas)) {
          const speed = parseInt($("#speed").value);
          if (speed <= 0) {  // slow down by skipping steps
            const skip = [1, 2, 10, 60][-speed];
            stepsPerFrame = (frameCount % skip) ? 0 : 1;
            frameCount += 1;
          } else if (speed > 0) { // speed up by making more steps per frame
            const interval = time - lastDrawTime;
            stepsPerFrame += interval < 20.0 ? 1 : -1;
            stepsPerFrame = Math.max(1, stepsPerFrame);
            stepsPerFrame = Math.min(stepsPerFrame, [1, 2, 4, Infinity][speed])
          }
          for (let i = 0; i < stepsPerFrame; ++i) {
            tf.tidy(() => {
                state.assign(model.execute(
                  { x: state,
                    fire_rate: tf.tensor(firingChance),
                    manual_noise: tf.randomNormal([1, h, w, ch-1], 0., 0.02)},
                  ['Identity']));
            });
          }
          if (stepsPerFrame > 0) {
            $("#ips").innerText = Math.round(lastStepCount/((time - lastStepTime)/1000.0));
            lastStepTime = time;
            lastStepCount = stepsPerFrame;
          }
        }
        const imageData = tf.tidy(() => {
            let rgbaBytes;
            let rgba;
            if (visibleChannel < 0) {
                const isGray = state.slice([0,0,0,0],[1, h, w, 1]).greater(0.1).toFloat();
                const isNotGray = tf.tensor(1.).sub(isGray);

                const bnwOrder = backgroundWhite ?  [isGray, isNotGray] : [isNotGray, isGray];
                let blackAndWhite = blackAndWhiteFull.mul(tf.concat(bnwOrder, 3));

                const grey = state.gather([0], 3).mul(255);
                const rgb = tf.gather(colorLookup,
                                      tf.argMax(
                                      tf.concat([
                  state.slice([0,0,0,ch-10],[1,h,w,10]),
                  blackAndWhite], 3), 3));

                rgba = tf.concat([rgb, grey], 3)
            } else {
                rgba = state.gather([visibleChannel, visibleChannel, visibleChannel], 3)
                  .pad([[0, 0], [0, 0], [0, 0], [0, 1]], 1).mul(255);
            }
            rgbaBytes = new Uint8ClampedArray(rgba.dataSync());

            return new ImageData(rgbaBytes, w, h);
        });
        render_ctx.putImageData(imageData, 0, 0);
        //const image = await createImageBitmap(imageData);
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = backgroundWhite ? "#ffffff" : "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(render_canvas, 0, 0, canvas.width, canvas.height);
        lastDrawTime = time;

        requestAnimationFrame(render);
      }
      requestAnimationFrame(render);
  }

  run();
}
