/**
 * Job Run History - Domino App
 * Frontend JavaScript for job history visualization
 * Uses React with React.createElement() for UI components (following Domino guidelines)
 */

// Shorthand for React.createElement
const h = React.createElement;

// State management
const state = {
  currentDays: 30,
  customDateRange: null, // [startDate, endDate] when using custom range
  chart: null,
  data: null
};

// Status color configuration (matches Highcharts)
const STATUS_COLORS = {
  succeeded: '#10b981',
  failed: '#ef4444',
  stopped: '#f59e0b',
  running: '#3b82f6',
  queued: '#8b5cf6'
};

// Date range presets (inspired by example_domino_frontend_code/TimePeriodSelector)
const DateRanges = {
  SEVEN_DAYS: '7days',
  THIRTY_DAYS: '30days',
  SIXTY_DAYS: '60days',
  NINETY_DAYS: '90days',
  CUSTOM: 'custom'
};

// Get base URL for API calls (handles Domino nginx proxy)
function getBaseUrl() {
  const path = window.location.pathname;
  const basePath = path.replace(/\/[^\/]*\.[^\/]*$/, '').replace(/\/$/, '');
  return basePath || '';
}

// Format date for display
function formatDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric',
    year: 'numeric'
  });
}

// Get month name helpers (from timePeriodSelectorUtil.ts)
function getCurrentMonthName() {
  return dayjs().format('MMM');
}

function getPreviousMonthName() {
  return dayjs().subtract(1, 'month').format('MMM');
}

// Update summary statistics
function updateStats(aggregated) {
  const totals = aggregated.reduce((acc, day) => {
    acc.succeeded += day.succeeded;
    acc.failed += day.failed;
    acc.stopped += day.stopped;
    acc.running += day.running;
    acc.queued += day.queued;
    return acc;
  }, { succeeded: 0, failed: 0, stopped: 0, running: 0, queued: 0 });

  const total = totals.succeeded + totals.failed + totals.stopped + totals.running + totals.queued;

  document.getElementById('stat-total').textContent = total.toLocaleString();
  document.getElementById('stat-succeeded').textContent = totals.succeeded.toLocaleString();
  document.getElementById('stat-failed').textContent = totals.failed.toLocaleString();
  document.getElementById('stat-stopped').textContent = totals.stopped.toLocaleString();
  document.getElementById('stat-running').textContent = totals.running.toLocaleString();
  document.getElementById('stat-queued').textContent = totals.queued.toLocaleString();
}

// Show loading state
function showLoading() {
  document.getElementById('loading-state').style.display = 'flex';
  document.getElementById('job-history-chart').style.display = 'none';
  document.getElementById('error-state').style.display = 'none';
}

// Show error state
function showError(message) {
  document.getElementById('loading-state').style.display = 'none';
  document.getElementById('job-history-chart').style.display = 'none';
  document.getElementById('error-state').style.display = 'flex';
  document.getElementById('error-message').textContent = message;
}

// Show chart
function showChart() {
  document.getElementById('loading-state').style.display = 'none';
  document.getElementById('job-history-chart').style.display = 'block';
  document.getElementById('error-state').style.display = 'none';
}

// Create or update the Highcharts bar chart
function renderChart(aggregated, startDate, endDate) {
  const startTime = new Date(startDate).getTime();
  const endTime = new Date(endDate).getTime();
  const daysDiff = Math.ceil((endTime - startTime) / (24 * 3600 * 1000));
  
  // Map data to timestamps
  const succeededSeries = aggregated.map(d => [new Date(d.date).getTime(), d.succeeded]);
  const failedSeries = aggregated.map(d => [new Date(d.date).getTime(), d.failed]);
  const stoppedSeries = aggregated.map(d => [new Date(d.date).getTime(), d.stopped]);
  const runningSeries = aggregated.map(d => [new Date(d.date).getTime(), d.running]);
  const queuedSeries = aggregated.map(d => [new Date(d.date).getTime(), d.queued]);

  const chartOptions = {
    chart: {
      type: 'column',
      backgroundColor: 'transparent',
      style: {
        fontFamily: 'Roboto, Lato, Helvetica Neue, sans-serif'
      },
      height: 400
    },
    title: {
      text: null
    },
    credits: {
      enabled: false
    },
    xAxis: {
      type: 'datetime',
      min: startTime,
      max: endTime,
      tickInterval: daysDiff <= 14 ? 24 * 3600 * 1000 : 7 * 24 * 3600 * 1000,
      labels: {
        format: '{value:%b %d}',
        style: {
          color: '#6b7280',
          fontSize: '12px'
        }
      },
      lineColor: '#e9edf5',
      tickColor: '#e9edf5',
      gridLineWidth: 0
    },
    yAxis: {
      min: 0,
      title: {
        text: 'Number of Jobs',
        style: {
          color: '#6b7280',
          fontSize: '12px',
          fontWeight: 'normal'
        }
      },
      labels: {
        style: {
          color: '#6b7280',
          fontSize: '12px'
        }
      },
      gridLineColor: '#e9edf5',
      gridLineDashStyle: 'Dash',
      allowDecimals: false
    },
    legend: {
      align: 'center',
      verticalAlign: 'bottom',
      layout: 'horizontal',
      itemStyle: {
        color: '#6b7280',
        fontSize: '12px',
        fontWeight: 'normal'
      },
      itemHoverStyle: {
        color: '#333333'
      },
      symbolRadius: 2,
      symbolHeight: 12,
      symbolWidth: 12
    },
    tooltip: {
      shared: true,
      backgroundColor: '#ffffff',
      borderColor: '#e9edf5',
      borderRadius: 8,
      shadow: {
        color: 'rgba(0,0,0,0.1)',
        offsetX: 0,
        offsetY: 4,
        opacity: 0.2,
        width: 12
      },
      style: {
        color: '#333333',
        fontSize: '13px'
      },
      xDateFormat: '%B %d, %Y',
      headerFormat: '<span style="font-weight: 600; font-size: 14px;">{point.key}</span><br/>',
      pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y}</b><br/>'
    },
    plotOptions: {
      column: {
        stacking: 'normal',
        borderRadius: 2,
        borderWidth: 0,
        pointPadding: 0.1,
        groupPadding: 0.05,
        pointRange: 24 * 3600 * 1000
      },
      series: {
        animation: {
          duration: 800
        }
      }
    },
    series: [{
      name: 'Succeeded',
      data: succeededSeries,
      color: STATUS_COLORS.succeeded
    }, {
      name: 'Failed',
      data: failedSeries,
      color: STATUS_COLORS.failed
    }, {
      name: 'Stopped',
      data: stoppedSeries,
      color: STATUS_COLORS.stopped
    }, {
      name: 'Running',
      data: runningSeries,
      color: STATUS_COLORS.running
    }, {
      name: 'Queued',
      data: queuedSeries,
      color: STATUS_COLORS.queued
    }]
  };

  if (state.chart) {
    state.chart.destroy();
  }

  state.chart = Highcharts.chart('job-history-chart', chartOptions);
  showChart();
}

// Load job history from API
async function loadJobHistory() {
  showLoading();
  
  const baseUrl = getBaseUrl();
  let url;
  
  if (state.customDateRange) {
    // Custom date range - calculate days between dates
    const [start, end] = state.customDateRange;
    const days = Math.ceil((end.valueOf() - start.valueOf()) / (24 * 3600 * 1000));
    url = `${baseUrl}/api/job-history?days=${days}`;
  } else {
    url = `${baseUrl}/api/job-history?days=${state.currentDays}`;
  }
  
  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    state.data = data;
    
    // Update the UI
    updateStats(data.aggregated);
    renderChart(data.aggregated, data.start_date, data.end_date);
    
    // Update the React component with new date range
    if (window.updateTimePeriodDisplay) {
      window.updateTimePeriodDisplay(data.start_date, data.end_date);
    }
    
  } catch (error) {
    console.error('Failed to load job history:', error);
    showError(error.message || 'Unable to fetch job history. Please try again.');
  }
}

/**
 * TimePeriodSelector React Component
 * Adapted from example_domino_frontend_code/frontend/apps/web/src/components/ui/TimePeriodSelector.tsx
 * Uses React.createElement() as per Domino app guidelines
 */
function TimePeriodSelector() {
  const [selectedRange, setSelectedRange] = React.useState(DateRanges.THIRTY_DAYS);
  const [dateRange, setDateRange] = React.useState(null);
  const [displayDates, setDisplayDates] = React.useState({ start: null, end: null });

  // Expose update function to window for external calls
  React.useEffect(() => {
    window.updateTimePeriodDisplay = (start, end) => {
      setDisplayDates({ start, end });
    };
  }, []);

  const handleRangeChange = (rangeKey) => {
    setSelectedRange(rangeKey);
    state.customDateRange = null;
    
    switch (rangeKey) {
      case DateRanges.SEVEN_DAYS:
        state.currentDays = 7;
        loadJobHistory();
        break;
      case DateRanges.THIRTY_DAYS:
        state.currentDays = 30;
        loadJobHistory();
        break;
      case DateRanges.SIXTY_DAYS:
        state.currentDays = 60;
        loadJobHistory();
        break;
      case DateRanges.NINETY_DAYS:
        state.currentDays = 90;
        loadJobHistory();
        break;
      case DateRanges.CUSTOM:
        // Don't load yet - wait for date picker selection
        break;
    }
  };

  const handleDatePickerChange = (dates) => {
    if (dates && dates.length === 2) {
      state.customDateRange = dates;
      setDateRange(dates);
      loadJobHistory();
    }
  };

  // Preset button style
  const buttonStyle = (isActive) => ({
    padding: '8px 16px',
    border: 'none',
    backgroundColor: isActive ? '#4568f6' : 'transparent',
    color: isActive ? 'white' : '#6b7280',
    fontSize: '14px',
    fontFamily: 'Roboto, sans-serif',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontWeight: isActive ? '500' : '400'
  });

  const containerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap'
  };

  const buttonGroupStyle = {
    display: 'flex',
    gap: '4px',
    backgroundColor: '#f8f9fc',
    padding: '4px',
    borderRadius: '6px'
  };

  const dateDisplayStyle = {
    fontSize: '14px',
    color: '#6b7280',
    padding: '8px 12px',
    backgroundColor: '#f8f9fc',
    borderRadius: '4px'
  };

  const labelStyle = {
    fontSize: '14px',
    color: '#6b7280',
    fontWeight: '500'
  };

  // Create preset buttons
  const presetButtons = [
    { key: DateRanges.SEVEN_DAYS, label: '7 Days' },
    { key: DateRanges.THIRTY_DAYS, label: '30 Days' },
    { key: DateRanges.SIXTY_DAYS, label: '60 Days' },
    { key: DateRanges.NINETY_DAYS, label: '90 Days' },
  ];

  return h('div', { style: containerStyle },
    // Label
    h('span', { style: labelStyle }, 'Time Range:'),
    
    // Preset buttons group
    h('div', { style: buttonGroupStyle },
      presetButtons.map(btn => 
        h('button', {
          key: btn.key,
          style: buttonStyle(selectedRange === btn.key),
          onClick: () => handleRangeChange(btn.key),
          onMouseOver: (e) => {
            if (selectedRange !== btn.key) {
              e.target.style.backgroundColor = 'rgba(69, 104, 246, 0.1)';
              e.target.style.color = '#4568f6';
            }
          },
          onMouseOut: (e) => {
            if (selectedRange !== btn.key) {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.color = '#6b7280';
            }
          }
        }, btn.label)
      ),
      // Custom button
      h('button', {
        style: buttonStyle(selectedRange === DateRanges.CUSTOM),
        onClick: () => handleRangeChange(DateRanges.CUSTOM),
        onMouseOver: (e) => {
          if (selectedRange !== DateRanges.CUSTOM) {
            e.target.style.backgroundColor = 'rgba(69, 104, 246, 0.1)';
            e.target.style.color = '#4568f6';
          }
        },
        onMouseOut: (e) => {
          if (selectedRange !== DateRanges.CUSTOM) {
            e.target.style.backgroundColor = 'transparent';
            e.target.style.color = '#6b7280';
          }
        }
      }, 'Custom')
    ),

    // Ant Design RangePicker (shown when Custom is selected)
    selectedRange === DateRanges.CUSTOM && h(antd.DatePicker.RangePicker, {
      onChange: handleDatePickerChange,
      format: 'MMM DD, YYYY',
      style: { marginLeft: '8px' },
      allowClear: false,
      disabledDate: (current) => current && current > dayjs().endOf('day')
    }),

    // Date range display
    displayDates.start && displayDates.end && h('span', { style: dateDisplayStyle },
      `${formatDate(displayDates.start)} – ${formatDate(displayDates.end)}`
    )
  );
}

// Initialize the application
function init() {
  // Set project info
  document.getElementById('project-info').textContent = 'Viewing job run statistics for this project';
  
  // Render React TimePeriodSelector component
  const selectorRoot = document.getElementById('time-period-selector-root');
  if (selectorRoot) {
    const root = ReactDOM.createRoot(selectorRoot);
    root.render(h(TimePeriodSelector));
  }
  
  // Load initial data
  loadJobHistory();
}

// Start the app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
