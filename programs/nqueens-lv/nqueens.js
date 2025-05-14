// N-Queens Simulation Logic (Las Vegas + Animation + Interactive Tracker)

const boardContainer = document.getElementById('board-container');
const statusDiv = document.getElementById('status');
const attemptsCounter = document.getElementById('attempts-counter');
const stepsCounter = document.getElementById('steps-counter');
const trackList = document.getElementById('track-list');
const speedSlider = document.getElementById('speed-slider');

let boardSize = 8;
let board = [];
let animating = false;
let animationDelay = 150; // ms, default
let attempts = 0;
let steps = 0;
let stepTracks = [];
let highlightStep = null;

function createBoard(size) {
  board = Array.from({ length: size }, () => Array(size).fill(0));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function renderBoard(highlight = null) {
  boardContainer.innerHTML = '';
  const table = document.createElement('div');
  table.className = `grid grid-cols-${boardSize} gap-0 border-2 border-gray-700 bg-gray-700`;
  table.style.width = 'min(90vw, 480px)';
  table.style.height = 'min(90vw, 480px)';
  for (let row = 0; row < boardSize; row++) {
    for (let col = 0; col < boardSize; col++) {
      const cell = document.createElement('div');
      const isDark = (row + col) % 2 === 1;
      let cellClass = `flex items-center justify-center text-2xl md:text-3xl lg:text-4xl select-none transition-all duration-150 ${isDark ? 'bg-gray-400' : 'bg-gray-100'}`;
      if (highlight && highlight.row === row && highlight.col === col) {
        cellClass += ' ring-4 ring-yellow-400';
      }
      if (highlightStep && highlightStep.row === row && highlightStep.col === col) {
        cellClass += ' ring-4 ring-pink-500';
      }
      cell.className = cellClass;
      cell.style.aspectRatio = '1';
      cell.style.width = '100%';
      cell.style.height = '100%';
      if (board[row][col] === 1) {
        cell.textContent = '♛';
        cell.classList.add('text-purple-700', 'font-bold');
      }
      table.appendChild(cell);
    }
  }
  boardContainer.appendChild(table);
}

function isSafe(row, col) {
  for (let i = 0; i < row; i++) {
    if (board[i][col] === 1) return false;
    if (col - (row - i) >= 0 && board[i][col - (row - i)] === 1) return false;
    if (col + (row - i) < boardSize && board[i][col + (row - i)] === 1) return false;
  }
  return true;
}

function getSafeColumns(row) {
  const safe = [];
  for (let col = 0; col < boardSize; col++) {
    if (isSafe(row, col)) safe.push(col);
  }
  return safe;
}

function updateCounters() {
  attemptsCounter.textContent = attempts;
  stepsCounter.textContent = steps;
}

function updateTrackList() {
  trackList.innerHTML = '';
  stepTracks.forEach((step, idx) => {
    const div = document.createElement('div');
    div.className = `cursor-pointer px-2 py-1 rounded mb-1 ${highlightStep && highlightStep.idx === idx ? 'bg-pink-100' : 'hover:bg-yellow-100'}`;
    div.textContent = step.text;
    div.onclick = () => {
      highlightStep = { ...step, idx };
      renderBoard({ row: step.row, col: step.col });
      updateTrackList();
    };
    trackList.appendChild(div);
  });
}

function updateProgressBar(row = 0) {
  const bar = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  if (!bar || !label) return;
  if (boardSize === 0) {
    bar.style.width = '0%';
    label.textContent = '';
    return;
  }
  const percent = Math.round((row / boardSize) * 100);
  bar.style.width = percent + '%';
  label.textContent = row > 0 ? `Row ${row} of ${boardSize}` : '';
}

function getSliderDelay() {
  // Invert: low value = slow (high delay), high value = fast (low delay)
  const min = 10, max = 250;
  const sliderValue = parseInt(speedSlider.value);
  return max + min - sliderValue;
}

async function lasVegasNQueensAnimate(maxAttempts = 1000) {
  animating = true;
  disableControls(true);
  attempts = 0;
  steps = 0;
  stepTracks = [];
  highlightStep = null;
  updateCounters();
  updateTrackList();
  updateProgressBar(0);
  let attempt = 0;
  while (attempt < maxAttempts) {
    attempt++;
    attempts = attempt;
    createBoard(boardSize);
    renderBoard();
    statusDiv.textContent = `Attempt ${attempt}...`;
    stepTracks.push({ text: `--- Attempt ${attempt} ---`, row: null, col: null });
    updateTrackList();
    updateProgressBar(0);
    await sleep(animationDelay);
    let success = true;
    for (let row = 0; row < boardSize; row++) {
      updateProgressBar(row);
      const safeCols = getSafeColumns(row);
      if (safeCols.length === 0) {
        success = false;
        stepTracks.push({ text: `Row ${row}: No safe columns, restart`, row, col: null });
        updateTrackList();
        break;
      }
      // Animate trying each safe column randomly
      const shuffled = safeCols.sort(() => Math.random() - 0.5);
      let placed = false;
      for (const col of shuffled) {
        steps++;
        updateCounters();
        stepTracks.push({ text: `Row ${row}: Try col ${col}`, row, col });
        updateTrackList();
        renderBoard({ row, col });
        animationDelay = getSliderDelay(); // Update delay in real time
        await sleep(animationDelay);
        if (isSafe(row, col)) {
          board[row][col] = 1;
          stepTracks.push({ text: `Row ${row}: Placed queen at col ${col}`, row, col });
          updateTrackList();
          renderBoard();
          animationDelay = getSliderDelay(); // Update delay in real time
          await sleep(animationDelay);
          placed = true;
          break;
        }
      }
      if (!placed) {
        success = false;
        break;
      }
    }
    updateProgressBar(success ? boardSize : 0);
    if (success) {
      statusDiv.textContent = `Solved in ${attempt} attempt${attempt > 1 ? 's' : ''}!`;
      renderBoard();
      animating = false;
      disableControls(false);
      updateCounters();
      updateTrackList();
      return true;
    } else {
      // Animate reset
      statusDiv.textContent = `Restarting (no solution in attempt ${attempt})...`;
      animationDelay = getSliderDelay(); // Update delay in real time
      await sleep(animationDelay * 1.5);
    }
  }
  updateProgressBar(0);
  statusDiv.textContent = `No solution found after ${maxAttempts} attempts.`;
  animating = false;
  disableControls(false);
  updateCounters();
  updateTrackList();
  updateProgressBar(0);
  return false;
}

function resetBoard() {
  createBoard(boardSize);
  renderBoard();
  statusDiv.textContent = '';
  attempts = 0;
  steps = 0;
  stepTracks = [];
  highlightStep = null;
  updateCounters();
  updateTrackList();
  updateProgressBar(0);
}

function disableControls(disable) {
  document.querySelectorAll('.board-size-btn, #solve-btn, #reset-btn, #speed-slider').forEach(btn => {
    btn.disabled = disable;
    if (disable) {
      btn.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
      btn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  });
}

speedSlider.addEventListener('input', () => {
  animationDelay = getSliderDelay();
});

// Event listeners

document.querySelectorAll('.board-size-btn').forEach(btn => {
  btn.addEventListener('click', e => {
    if (animating) return;
    boardSize = parseInt(btn.getAttribute('data-size'));
    resetBoard();
  });
});

document.getElementById('solve-btn').addEventListener('click', async () => {
  if (animating) return;
  await lasVegasNQueensAnimate();
});

document.getElementById('reset-btn').addEventListener('click', () => {
  if (animating) return;
  resetBoard();
});

// Initialize default board
resetBoard();