// static/app.js - complete working version
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileList = document.getElementById('file-list');
    const videoPlayer = document.getElementById('video-player');
    const videoSource = document.getElementById('video-source');
    const timelineMarker = document.getElementById('timeline-marker');
    const windowInfo = document.getElementById('window-info');
    const keyboardDisplay = document.getElementById('keyboard-display');
    const mouseDisplay = document.getElementById('mouse-display');
    const mouseCursor = document.getElementById('mouse-cursor');
    const timeline = document.getElementById('timeline');
    
    // State
    let currentFile = null;
    let currentData = {
        keyboard: [],
        mouse: [],
        screen: [],
        window: [],
        "keyboard/state": [],
        "mouse/state": []
    };
    let metadata = null;
    let basePtsTime = 0;
    let lastLoadedTime = null;
    let isLoading = false;
    let timelineControls = null;

    let isPlaying = false;
    let lastRenderTime = 0;
    let animationFrameId = null;
    
    // Constants
    const DATA_WINDOW_SIZE = 10_000_000_000; // 10 seconds in nanoseconds
    const SEEK_BUFFER = 2_000_000_000; // 2 seconds buffer before current position
    
    // Fetch list of available file pairs
    async function fetchFilePairs() {
        try {
            const response = await fetch('/api/file_pairs');
            const data = await response.json();
            
            console.log("Available file pairs:", data);
            
            fileList.innerHTML = '';
            data.forEach(pair => {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.textContent = pair.basename;
                item.addEventListener('click', () => loadFilePair(pair));
                fileList.appendChild(item);
            });
        } catch (error) {
            console.error("Error fetching file pairs:", error);
            fileList.innerHTML = '<div class="error">Error loading files. Check console.</div>';
        }
    }
    
    // Load a specific MCAP+MKV pair
    async function loadFilePair(pair) {
        try {
            console.log("Loading file pair:", pair);
            currentFile = pair;
            
            // Update UI to show selected file
            document.querySelectorAll('.file-item').forEach(item => {
                item.classList.remove('active');
                if (item.textContent === pair.basename) {
                    item.classList.add('active');
                }
            });
            
            // Clear previous data
            for (let topic in currentData) {
                currentData[topic] = [];
            }
            
            // Set loading state
            setLoadingState(true);
            
            // Set the video source
            console.log(`Setting video source to: /video/${pair.mkv_file}`);
            videoSource.src = `/video/${pair.mkv_file}`;
            videoPlayer.load();
            console.log("Video source set successfully");
            
            // Fetch MCAP metadata
            console.log(`Fetching MCAP metadata: /api/mcap_metadata/${pair.mcap_file}`);
            const metaResponse = await fetch(`/api/mcap_metadata/${pair.mcap_file}`);
            if (!metaResponse.ok) {
                throw new Error(`HTTP error! status: ${metaResponse.status}`);
            }
            
            metadata = await metaResponse.json();
            console.log("MCAP metadata loaded:", metadata);
            
            // Initialize with data from the beginning
            await loadDataForTimeRange(metadata.start_time, null);
            
            // Process the screen topics to find base time
            if (currentData.screen && currentData.screen.length > 0) {
                const firstScreenEvent = currentData.screen[0];
                console.log("First screen event:", firstScreenEvent);
                console.log("First screen event timestamp:", firstScreenEvent.timestamp);
                basePtsTime = firstScreenEvent.timestamp || 0;
                console.log("Base PTS time:", basePtsTime);
            } else {
                console.warn("No screen events found in MCAP data");
                basePtsTime = 0;
            }
            
            // Initialize UI with data
            renderInitialState();
            
            // Add video event listeners
            setupVideoSync();
            
            // Setup enhanced timeline (must be after data is loaded)
            setupEnhancedTimeline();

            // Stop any existing visualization loop
            stopVisualizationLoop();

            // Start visualization loop if video is already playing
            if (!videoPlayer.paused) {
                isPlaying = true;
                startVisualizationLoop();
            }
            
            // Update timeline visualization
            updateTimelineLoadedRegions();
            
        } catch (error) {
            console.error("Error loading file pair:", error);
            alert(`Error loading file: ${error.message}`);
        } finally {
            setLoadingState(false);
        }
    }
    
    // Set loading state and update UI accordingly
    function setLoadingState(isLoading) {
        const loadingIndicator = document.getElementById('loading-indicator') || 
                                 document.createElement('div');
        
        if (isLoading) {
            loadingIndicator.id = 'loading-indicator';
            loadingIndicator.textContent = 'Loading data...';
            loadingIndicator.style.position = 'fixed';
            loadingIndicator.style.top = '10px';
            loadingIndicator.style.right = '10px';
            loadingIndicator.style.padding = '10px';
            loadingIndicator.style.backgroundColor = '#ffe082';
            loadingIndicator.style.zIndex = '1000';
            loadingIndicator.style.borderRadius = '4px';
            
            document.body.appendChild(loadingIndicator);
        } else if (document.getElementById('loading-indicator')) {
            document.body.removeChild(loadingIndicator);
        }
    }
    
    // Load MCAP data for a specific time range
    // NOTE: starTime and endTime are pts time in nanoseconds.
    async function loadDataForTimeRange(startTime, endTime) {
        if (!currentFile || isLoading) return;
        
        // Avoid duplicate loads for the same time
        if (lastLoadedTime && Math.abs(lastLoadedTime - startTime) < SEEK_BUFFER) {
            return;
        }
        
        isLoading = true;
        setLoadingState(true);
        
        try {
            const url = new URL(`/api/mcap_data/${currentFile.mcap_file}`, window.location.origin);
            url.searchParams.append('start_time', startTime);
            if (endTime) url.searchParams.append('end_time', endTime);
            url.searchParams.append('window_size', DATA_WINDOW_SIZE);
            
            console.log(`Loading data for time range: ${startTime} to ${endTime || startTime + DATA_WINDOW_SIZE}`);
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const newData = await response.json();
            
            // Merge new data with existing data
            for (let topic in newData) {
                // Clear old data for this time range
                currentData[topic] = currentData[topic].filter(msg => 
                    msg.timestamp < startTime || 
                    (endTime && msg.timestamp > endTime)
                );
                
                // Add new data
                currentData[topic] = [...currentData[topic], ...newData[topic]];
                
                // Sort by timestamp
                currentData[topic].sort((a, b) => a.timestamp - b.timestamp);
            }
            
            lastLoadedTime = startTime;
            console.log("Data loaded and merged successfully");
            
            // Update timeline visualization after loading data
            updateTimelineLoadedRegions();
            
        } catch (error) {
            console.error("Error loading data for time range:", error);
        } finally {
            isLoading = false;
            setLoadingState(false);
        }
    }
    
    // Set up video synchronization with MCAP data
    function setupVideoSync() {
        // Remove previous event listeners
        videoPlayer.onplay = () => {
            isPlaying = true;
            startVisualizationLoop();
        };
        
        videoPlayer.onpause = () => {
            isPlaying = false;
            stopVisualizationLoop();
        };
        
        videoPlayer.onseeking = handleSeeking;
        
        // Remove the timeupdate listener since we'll use requestAnimationFrame
        videoPlayer.ontimeupdate = null;
        
        console.log("Video sync setup complete");
    }

    function startVisualizationLoop() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        
        function render(timestamp) {
            // Limit updates to ~60fps
            if (timestamp - lastRenderTime >= 16.67) { // roughly 60fps (1000ms / 60)
                updateVisualizations();
                checkDataLoading();
                lastRenderTime = timestamp;
            }
            
            if (isPlaying) {
                animationFrameId = requestAnimationFrame(render);
            }
        }
        
        animationFrameId = requestAnimationFrame(render);
    }
    
    function stopVisualizationLoop() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }
    
    // Add this function to handle data loading checks
    function checkDataLoading() {
        if (!metadata) return;
        
        const videoTime = videoPlayer.currentTime || 0;
        const currentTimeNs = basePtsTime + (videoTime * 1000000000);
        
        // If we're getting close to the end of our loaded data window, load more
        if (lastLoadedTime && currentTimeNs > lastLoadedTime + (DATA_WINDOW_SIZE * 0.7)) {
            loadDataForTimeRange(currentTimeNs - SEEK_BUFFER, currentTimeNs + DATA_WINDOW_SIZE);
        }
    }
    
    // Handle seeking in the video
    function handleSeeking() {
        const videoTime = videoPlayer.currentTime || 0;
        const seekTimeNs = basePtsTime + (videoTime * 1000000000);
        
        console.log(`Seeking to video time ${videoTime}s, MCAP time ${seekTimeNs}ns`);
        
        // Load data for the new position
        loadDataForTimeRange(seekTimeNs - SEEK_BUFFER, seekTimeNs + DATA_WINDOW_SIZE)
            .then(() => {
                // Update visualizations immediately after data is loaded
                updateVisualizations();
            });
            
        // Ensure visualization loop is in correct state
        if (isPlaying) {
            startVisualizationLoop();
        }
    }
    
    // Update visualizations based on current video time
    function updateVisualizations() {
        if (!currentData) {
            console.warn("No data available for visualization");
            return;
        }
        
        // If video isn't playing yet, use 0 as the current time
        const videoTime = videoPlayer.currentTime || 0;
        const currentTimeNs = basePtsTime + (videoTime * 1000000000);
        
        // Update timeline marker
        const percentage = videoPlayer.duration ? (videoTime / videoPlayer.duration) * 100 : 0;
        timelineMarker.style.left = `${percentage}%`;
        
        // Find the most recent events for each topic that occurred before currentTimeNs
        updateWindowInfo(currentTimeNs);
        updateKeyboardState(currentTimeNs);
        updateMouseState(currentTimeNs);
    }
    
    // Update window information display
    function updateWindowInfo(currentTimeNs) {
        if (!currentData.window || currentData.window.length === 0) {
            windowInfo.innerHTML = 'No window data available';
            return;
        }
        
        const event = findLastEventBeforeTime(currentData.window, currentTimeNs);
        if (event) {
            windowInfo.innerHTML = `
                <p>Title: ${event.title || 'Unknown'}</p>
                <p>Position: X=${event.rect?.[0] || 0}, Y=${event.rect?.[1] || 0}</p>
                <p>Size: W=${(event.rect?.[2] || 0) - (event.rect?.[0] || 0)}, 
                         H=${(event.rect?.[3] || 0) - (event.rect?.[1] || 0)}</p>
            `;
        } else {
            windowInfo.innerHTML = 'No window events at current time';
        }
    }
    
    // Update keyboard state display
    function updateKeyboardState(currentTimeNs) {
        const keyboardState = currentData['keyboard/state'] || [];
        
        if (keyboardState.length === 0) {
            keyboardDisplay.innerHTML = 'No keyboard state data available';
            return;
        }
        
        const event = findLastEventBeforeTime(keyboardState, currentTimeNs);
        if (event) {
            // Clear previous state
            keyboardDisplay.innerHTML = '';
            
            // Check keyboard events to determine currently pressed keys
            const pressedKeys = new Set(event.buttons || []);
            
            // Create key elements
            for (const key of pressedKeys) {
                const keyElem = document.createElement('div');
                keyElem.className = 'key pressed';
                keyElem.textContent = getKeyName(key);
                keyboardDisplay.appendChild(keyElem);
            }
            
            if (pressedKeys.size === 0) {
                keyboardDisplay.innerHTML = '<div>No keys pressed</div>';
            }
        } else {
            keyboardDisplay.innerHTML = 'No keyboard events at current time';
        }
    }
    
    // Update mouse state display
    function updateMouseState(currentTimeNs) {
        const mouseState = currentData['mouse/state'] || [];
        
        if (mouseState.length === 0) {
            mouseDisplay.innerHTML = '<div>No mouse state data available</div>';
            mouseCursor.style.display = 'none';
            return;
        }
        
        const event = findLastEventBeforeTime(mouseState, currentTimeNs);
        if (event) {
            // Make sure cursor is visible
            mouseCursor.style.display = 'block';
            
            // Scale mouse position to fit in our display area
            const displayWidth = mouseDisplay.clientWidth;
            const displayHeight = mouseDisplay.clientHeight;
            
            // Assume screen dimensions - ideally these would come from the data
            const screenWidth = 1920;
            const screenHeight = 1080;
            
            const x = ((event.x || 0) / screenWidth) * displayWidth;
            const y = ((event.y || 0) / screenHeight) * displayHeight;
            
            mouseCursor.style.left = `${x}px`;
            mouseCursor.style.top = `${y}px`;
            
            // Highlight if buttons pressed
            if (event.buttons && event.buttons.length > 0) {
                mouseCursor.style.backgroundColor = 'blue';
                mouseCursor.style.width = '12px';
                mouseCursor.style.height = '12px';
            } else {
                mouseCursor.style.backgroundColor = 'red';
                mouseCursor.style.width = '10px';
                mouseCursor.style.height = '10px';
            }
        } else {
            mouseCursor.style.display = 'none';
        }
    }
    
    // Utility function to find the most recent event before a given time
    function findLastEventBeforeTime(events, time) {
        let lastEvent = null;
        
        for (const event of events) {
            if (event.timestamp <= time) {
                lastEvent = event;
            } else {
                // Assuming events are sorted by timestamp
                break;
            }
        }
        
        return lastEvent;
    }
    
    // Render initial state of visualizations
    function renderInitialState() {
        windowInfo.innerHTML = 'Waiting for data...';
        keyboardDisplay.innerHTML = 'Waiting for data...';
        mouseCursor.style.display = 'none';
    }
    
    // Helper function to convert virtual key codes to names
    function getKeyName(vk) {
        const keyMap = {
            8: 'BKSP',
            9: 'TAB',
            13: 'ENTER',
            16: 'SHIFT',
            17: 'CTRL',
            18: 'ALT',
            19: 'PAUSE',
            20: 'CAPS',
            27: 'ESC',
            32: 'SPACE',
            33: 'PGUP',
            34: 'PGDN',
            35: 'END',
            36: 'HOME',
            37: '←',
            38: '↑',
            39: '→',
            40: '↓',
            44: 'PRTSC',
            45: 'INS',
            46: 'DEL',
            48: '0',
            49: '1',
            50: '2',
            51: '3',
            52: '4',
            53: '5',
            54: '6',
            55: '7',
            56: '8',
            57: '9',
            65: 'A',
            66: 'B',
            67: 'C',
            68: 'D',
            69: 'E',
            70: 'F',
            71: 'G',
            72: 'H',
            73: 'I',
            74: 'J',
            75: 'K',
            76: 'L',
            77: 'M',
            78: 'N',
            79: 'O',
            80: 'P',
            81: 'Q',
            82: 'R',
            83: 'S',
            84: 'T',
            85: 'U',
            86: 'V',
            87: 'W',
            88: 'X',
            89: 'Y',
            90: 'Z',
            91: 'WIN',
            92: 'WIN',
            93: 'MENU',
            96: 'NUM0',
            97: 'NUM1',
            98: 'NUM2',
            99: 'NUM3',
            100: 'NUM4',
            101: 'NUM5',
            102: 'NUM6',
            103: 'NUM7',
            104: 'NUM8',
            105: 'NUM9',
            106: 'NUM*',
            107: 'NUM+',
            109: 'NUM-',
            110: 'NUM.',
            111: 'NUM/',
            112: 'F1',
            113: 'F2',
            114: 'F3',
            115: 'F4',
            116: 'F5',
            117: 'F6',
            118: 'F7',
            119: 'F8',
            120: 'F9',
            121: 'F10',
            122: 'F11',
            123: 'F12',
            144: 'NUMLOCK',
            145: 'SCRLOCK',
            160: 'SHIFT',
            161: 'SHIFT',
            162: 'CTRL',
            163: 'CTRL',
            164: 'ALT',
            165: 'ALT',
            186: ';',
            187: '=',
            188: ',',
            189: '-',
            190: '.',
            191: '/',
            192: '`',
            219: '[',
            220: '\\',
            221: ']',
            222: '\'',
            223: '`'
        };
        
        return keyMap[vk] || `VK${vk}`;
    }
    
    // Enhanced timeline functionality
    function setupEnhancedTimeline() {
        // Add a seekable-time indicator to show loaded data ranges
        let seekableTime = document.getElementById('seekable-time');
        if (!seekableTime) {
            seekableTime = document.createElement('div');
            seekableTime.id = 'seekable-time';
            seekableTime.className = 'seekable-time';
            timeline.appendChild(seekableTime);
        }
        
        // Handle clicking on the timeline to seek
        timeline.addEventListener('click', (e) => {
            if (!videoPlayer.duration) return;
            
            const rect = timeline.getBoundingClientRect();
            const position = (e.clientX - rect.left) / rect.width;
            const seekTime = videoPlayer.duration * position;
            
            // Seek the video
            videoPlayer.currentTime = seekTime;
        });
        
        // Update the seekable range indicator
        updateSeekableRange();
        
        console.log("Enhanced timeline setup complete");
    }
    
    // Update the seekable range indicator
    function updateSeekableRange() {
        if (!videoPlayer.duration || !metadata) return;
        
        const seekableTime = document.getElementById('seekable-time');
        if (!seekableTime) return;
        
        // Calculate the loaded data range as a percentage of the video duration
        const videoStartTimeNs = basePtsTime;
        const videoEndTimeNs = basePtsTime + (videoPlayer.duration * 1000000000);
        const loadedStartTimeNs = lastLoadedTime || videoStartTimeNs;
        const loadedEndTimeNs = loadedStartTimeNs + DATA_WINDOW_SIZE;
        
        // Convert to percentages
        const startPercent = ((loadedStartTimeNs - videoStartTimeNs) / (videoEndTimeNs - videoStartTimeNs)) * 100;
        const endPercent = ((loadedEndTimeNs - videoStartTimeNs) / (videoEndTimeNs - videoStartTimeNs)) * 100;
        
        // Update the seekable-time element
        seekableTime.style.left = `${Math.max(0, startPercent)}%`;
        seekableTime.style.width = `${Math.min(100, endPercent) - Math.max(0, startPercent)}%`;
    }
    
    // Timeline data loading visualization
    function updateTimelineLoadedRegions() {
        // Remove existing loaded regions
        document.querySelectorAll('.timeline-loaded').forEach(el => el.remove());
        
        if (!metadata || !videoPlayer.duration) return;
        
        // Get start and end time of loaded data in video seconds
        if (!lastLoadedTime) return;
        
        const videoStart = (lastLoadedTime - basePtsTime) / 1000000000;
        const videoEnd = (lastLoadedTime + DATA_WINDOW_SIZE - basePtsTime) / 1000000000;
        
        // Calculate position as percentage of video duration
        const startPercent = (videoStart / videoPlayer.duration) * 100;
        const widthPercent = ((videoEnd - videoStart) / videoPlayer.duration) * 100;
        
        // Create loaded region indicator
        const loadedRegion = document.createElement('div');
        loadedRegion.className = 'timeline-loaded';
        loadedRegion.style.left = `${startPercent}%`;
        loadedRegion.style.width = `${widthPercent}%`;
        
        timeline.appendChild(loadedRegion);
    }
    
    // Initialize the application
    fetchFilePairs();
});