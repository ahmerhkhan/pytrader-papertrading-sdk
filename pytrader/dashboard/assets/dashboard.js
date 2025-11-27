const selectors = {
  botMeta: document.getElementById('botMeta'),
  connectionStatus: document.getElementById('connectionStatus'),
  equityValue: document.getElementById('equityValue'),
  cashValue: document.getElementById('cashValue'),
  investedValue: document.getElementById('investedValue'),
  sessionReturn: document.getElementById('sessionReturn'),
  sharpeRatio: document.getElementById('sharpeRatio'),
  sortinoRatio: document.getElementById('sortinoRatio'),
  maxDrawdown: document.getElementById('maxDrawdown'),
  lastUpdated: document.getElementById('lastUpdated'),
  positionsCount: document.getElementById('positionsCount'),
  tradesCount: document.getElementById('tradesCount'),
  positionsTable: document.getElementById('positionsTable'),
  tradesTable: document.getElementById('tradesTable'),
  chartCanvas: document.getElementById('equityChart'),
};

let equityChart;
let reconnectTimeout;

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'PKR',
  maximumFractionDigits: 0,
});

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return currencyFormatter.format(value);
}

function formatPercent(value, fractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(fractionDigits)}%`;
}

function formatNumber(value, fractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(fractionDigits);
}

function updateMeta(snapshot) {
  if (!snapshot || !selectors.botMeta) {
    return;
  }
  const symbols = snapshot.bot.symbols.join(', ');
  selectors.botMeta.textContent = `${snapshot.bot.id} · ${symbols}`;
  if (selectors.lastUpdated) {
    selectors.lastUpdated.textContent = new Date(snapshot.updated_at).toLocaleString();
  }
}

function updateStats(snapshot) {
  selectors.equityValue.textContent = formatCurrency(snapshot.equity);
  selectors.cashValue.textContent = formatCurrency(snapshot.cash);
  selectors.investedValue.textContent = formatCurrency(snapshot.positions_value);

  const metrics = snapshot.metrics || {};
  const sessionReturn = metrics.session_return_pct ?? null;
  selectors.sessionReturn.textContent = formatPercent(sessionReturn);
  selectors.sessionReturn.classList.toggle('positive', sessionReturn > 0);
  selectors.sessionReturn.classList.toggle('negative', sessionReturn < 0);

  const sharpe = metrics.sharpe_ratio ?? metrics.sharpe ?? null;
  const sortino = metrics.sortino_ratio ?? metrics.sortino ?? null;
  const maxDrawdown = metrics.max_drawdown_pct ?? null;

  selectors.sharpeRatio.textContent = formatNumber(sharpe);
  selectors.sortinoRatio.textContent = formatNumber(sortino);
  selectors.maxDrawdown.textContent = formatPercent(maxDrawdown);
  selectors.maxDrawdown.classList.toggle('negative', (maxDrawdown ?? 0) < 0);
}

function updatePositions(positions = []) {
  selectors.positionsCount.textContent = `${positions.length} positions`;
  if (!positions.length) {
    selectors.positionsTable.innerHTML =
      '<tr><td colspan="6" class="muted centered">No open positions</td></tr>';
    return;
  }

  selectors.positionsTable.innerHTML = positions
    .map(
      (pos) => `
    <tr>
      <td>${pos.symbol}</td>
      <td>${pos.qty}</td>
      <td>${formatCurrency(pos.avg_cost)}</td>
      <td>${formatCurrency(pos.current_price)}</td>
      <td>${formatCurrency(pos.market_value)}</td>
      <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
        ${formatCurrency(pos.unrealized_pnl)}
      </td>
    </tr>`
    )
    .join('');
}

function updateTrades(trades = []) {
  selectors.tradesCount.textContent = `${trades.length} trades`;
  if (!trades.length) {
    selectors.tradesTable.innerHTML =
      '<tr><td colspan="6" class="muted centered">No trades yet</td></tr>';
    return;
  }

  selectors.tradesTable.innerHTML = trades
    .slice(0, 15)
    .map(
      (trade) => `
    <tr>
      <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
      <td>${trade.symbol}</td>
      <td class="${trade.side === 'BUY' ? 'positive' : 'negative'}">${trade.side}</td>
      <td>${trade.quantity}</td>
      <td>${formatCurrency(trade.price)}</td>
      <td class="${trade.pnl_realized >= 0 ? 'positive' : 'negative'}">
        ${formatCurrency(trade.pnl_realized)}
      </td>
    </tr>`
    )
    .join('');
}

function updateChart(history = []) {
  if (!selectors.chartCanvas) {
    return;
  }
  const labels = history.map((point) => new Date(point.timestamp).toLocaleTimeString());
  const values = history.map((point) => point.equity);

  if (!equityChart) {
    const context = selectors.chartCanvas.getContext('2d');
    equityChart = new Chart(context, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Equity',
            data: values,
            borderColor: 'rgba(76, 201, 240, 1)',
            backgroundColor: 'rgba(76, 201, 240, 0.1)',
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            ticks: { color: 'rgba(255,255,255,0.45)' },
            grid: { display: false },
          },
          y: {
            ticks: {
              color: 'rgba(255,255,255,0.45)',
              callback: (value) => formatCurrency(value),
            },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
        },
      },
    });
    return;
  }

  equityChart.data.labels = labels;
  equityChart.data.datasets[0].data = values;
  equityChart.update('none');
}

function setConnectionState(state) {
  if (!selectors.connectionStatus) {
    return;
  }
  selectors.connectionStatus.textContent = state.label;
  selectors.connectionStatus.className = `badge ${state.className}`;
}

function applySnapshot(snapshot) {
  if (!snapshot) {
    return;
  }
  updateMeta(snapshot);
  updateStats(snapshot);
  updatePositions(snapshot.positions);
  updateTrades(snapshot.recent_trades);
  updateChart(snapshot.equity_history);
  setConnectionState({ label: 'Live', className: 'badge--live' });
}

async function fetchInitialState() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Unable to load dashboard state');
    }
    const payload = await response.json();
    applySnapshot(payload);
  } catch (error) {
    console.warn('[dashboard] failed to boot state', error);
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.addEventListener('open', () => {
    setConnectionState({ label: 'Live', className: 'badge--live' });
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  });

  ws.addEventListener('close', () => {
    setConnectionState({ label: 'Reconnecting…', className: 'badge--idle' });
    reconnectTimeout = setTimeout(connectWebSocket, 2_000);
  });

  ws.addEventListener('message', (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.payload) {
        applySnapshot(message.payload);
      }
    } catch (error) {
      console.warn('[dashboard] failed to parse message', error);
    }
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  setConnectionState({ label: 'Connecting', className: 'badge--idle' });
  await fetchInitialState();
  connectWebSocket();
});

