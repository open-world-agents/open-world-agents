// static/app.js - complete working version
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileList = document.getElementById('file-list');
    const uploadedFileList = document.getElementById('uploaded-file-list');
    const videoPlayer = document.getElementById('video-player');
    const videoSource = document.getElementById('video-source');
    const timelineMarker = document.getElementById('timeline-marker');
    const windowInfo = document.getElementById('window-info');
    const keyboardDisplay = document.getElementById('keyboard-display');
    const mouseDisplay = document.getElementById('mouse-display');
    let mouseCursor = document.getElementById('mouse-cursor');
    const timeline = document.getElementById('timeline');
    const repoId = window.APP_CONFIG.repoId;

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
    let uploadedFiles = []; // Store uploaded files during session

    let isPlaying = false;
    let lastRenderTime = 0;
    let animationFrameId = null;

    // Constants
    const DATA_WINDOW_SIZE = 10_000_000_000; // 10 seconds in nanoseconds
    const SEEK_BUFFER = 2_000_000_000; // 2 seconds buffer before current position

    // Fetch list of available file pairs
    async function fetchFilePairs() {
        try {
            const response = await fetch(`/api/list_files?repo_id=${repoId}`);
            const data = await response.json();

            console.log("Available file pairs:", data);

            // Check if there's a currently selected file before clearing the list
            const currentlySelectedId = Array.from(document.querySelectorAll('.file-item.active'))
                .find(item => item.parentElement === fileList)?.dataset.uniqueId;

            fileList.innerHTML = '';
            data.forEach((pair, index) => {
                const item = document.createElement('div');
                item.className = 'file-item';

                // Display original filename if available, otherwise use the basename
                const displayName = pair.original_basename || pair.basename;
                item.textContent = displayName;

                // Store the unique filename as a data attribute for selection
                item.dataset.uniqueId = pair.basename;

                item.addEventListener('click', () => loadFilePair(pair));
                fileList.appendChild(item);

                // Only add auto-select class to first item if no file was previously selected
                if (index === 0 && !currentlySelectedId) {
                    item.classList.add('auto-select');
                }
            });

            // Automatically load the first file only if no file is currently selected
            if (data.length > 0 && !currentlySelectedId && !currentFile) {
                console.log("No file is currently selected. Auto-selecting the first file:", data[0].basename);
                loadFilePair(data[0]);
            }
        } catch (error) {
            console.error("Error fetching file pairs:", error);
            fileList.innerHTML = '<div class="error">Error loading files. Check console.</div>';
        }
    }

    // Function to update uploaded files list
    function updateUploadedFilesList() {
        uploadedFileList.innerHTML = '';

        if (uploadedFiles.length === 0) {
            uploadedFileList.innerHTML = '<div class="file-item-placeholder">No uploaded files yet</div>';
            return;
        }

        uploadedFiles.forEach(pair => {
            const item = document.createElement('div');
            item.className = 'file-item';

            // Use original filename for display if available
            const displayName = pair.original_basename || pair.basename;
            item.textContent = displayName;

            // Store the unique filename as a data attribute for selection
            item.dataset.uniqueId = pair.basename;

            item.addEventListener('click', () => loadFilePair(pair));
            uploadedFileList.appendChild(item);
        });
    }

    // Load a specific MCAP+MKV pair
    async function loadFilePair(pair) {
        try {
            console.log("Loading file pair:", pair);
            currentFile = pair;

            // Update UI to show selected file - use the unique ID to select only the correct file
            document.querySelectorAll('.file-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.uniqueId === pair.basename) {
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
            if (pair.local) {
                console.log(`Setting video source to: /files/${pair.url_mkv}`);
                videoSource.src = `/files/${pair.url_mkv}`;
            } else {
                console.log(`Setting video source to: ${pair.url_mkv}`);
                videoSource.src = pair.url_mkv;
            }
            videoPlayer.load();
            console.log("Video source set successfully");

            // Fetch MCAP metadata
            console.log(`Fetching MCAP metadata: /api/mcap_metadata?mcap_filename=${pair.url_mcap}&local=${pair.local}`);
            const metaResponse = await fetch(`/api/mcap_metadata?mcap_filename=${pair.url_mcap}&local=${pair.local}`);
            if (!metaResponse.ok) {
                throw new Error(`HTTP error! status: ${metaResponse.status}`);
            }

            metadata = await metaResponse.json();
            console.log("MCAP metadata loaded:", metadata);

            await updateMcapInfo(pair);

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

    async function updateMcapInfo(pair) {
        try {
            const response = await fetch(`/api/mcap_info?mcap_filename=${pair.url_mcap}&local=${pair.local}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const mcapInfo = (await response.json())["info"];

            // Replace literal "\n" with actual newlines and "\t" with actual tabs
            const formattedInfo = mcapInfo
                .replace(/\\n/g, '\n')
                .replace(/\\t/g, '\t');

            document.getElementById('mcap-info').innerHTML = `<pre>${formattedInfo}</pre>`;
        } catch (error) {
            console.error('Error fetching MCAP info:', error);
            document.getElementById('mcap-info').innerHTML =
                `<pre>Error loading MCAP info: ${error.message}</pre>`;
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
            const url = new URL(`/api/mcap_data?mcap_filename=${currentFile.url_mcap}&local=${currentFile.local}`, window.location.origin);
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

    // Add this function to handle data loading checks.
    // TODO: pause playback before we ensure the data is loaded.
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
        const keyboardEvents = currentData['keyboard'] || [];

        if (keyboardState.length === 0 && keyboardEvents.length === 0) {
            keyboardDisplay.innerHTML = 'No keyboard data available';
            return;
        }

        // Get current state
        const stateEvent = findLastEventBeforeTime(keyboardState, currentTimeNs);
        // Get recent events (within last 500ms for visual feedback)
        const recentEvents = keyboardEvents.filter(event =>
            event.timestamp > currentTimeNs - 500000000 &&
            event.timestamp <= currentTimeNs
        );

        // Clear previous state
        keyboardDisplay.innerHTML = '';

        // Create state section
        const stateSection = document.createElement('div');
        stateSection.className = 'keyboard-state';
        stateSection.innerHTML = '<h4>Current State</h4>';

        if (stateEvent) {
            const pressedKeys = new Set((stateEvent.buttons || [])
                .filter(vk => ![1, 2, 4].includes(vk)));

            for (const key of pressedKeys) {
                const keyElem = document.createElement('div');
                keyElem.className = 'key pressed';
                keyElem.textContent = getKeyName(key);
                stateSection.appendChild(keyElem);
            }
            if (pressedKeys.size === 0) {
                stateSection.innerHTML += '<div>No keys pressed</div>';
            }
        }

        // Create events section
        const eventsSection = document.createElement('div');
        eventsSection.className = 'keyboard-events';
        eventsSection.innerHTML = '<h4>Recent Events</h4>';

        if (recentEvents.length > 0) {
            recentEvents.forEach(event => {
                const eventElem = document.createElement('div');
                eventElem.className = `key-event ${event.event_type}`;
                eventElem.textContent = `${getKeyName(event.vk)} (${event.event_type})`;
                eventsSection.appendChild(eventElem);
            });
        } else {
            eventsSection.innerHTML += '<div>No recent events</div>';
        }

        keyboardDisplay.appendChild(stateSection);
        keyboardDisplay.appendChild(eventsSection);
    }

    // Update mouse state display
    function updateMouseState(currentTimeNs) {
        const mouseState = currentData['mouse/state'] || [];
        const mouseEvents = currentData['mouse'] || [];

        if (mouseState.length === 0 && mouseEvents.length === 0) {
            mouseDisplay.innerHTML = '<div>No mouse data available</div>';
            mouseCursor.style.display = 'none';
            return;
        }

        // Get current state and most recent event
        const stateEvent = findLastEventBeforeTime(mouseState, currentTimeNs);
        const lastEvent = findLastEventBeforeTime(mouseEvents, currentTimeNs);

        // Use state event for position, but if there's a more recent mouse movement event, use that
        let currentEvent = stateEvent;
        if (lastEvent && (!stateEvent || lastEvent.timestamp > stateEvent.timestamp)) {
            currentEvent = lastEvent;
        }

        // Get recent click and scroll events (within last 1000ms)
        const recentSpecialEvents = mouseEvents.filter(event =>
            (event.event_type === 'click' || event.event_type === 'scroll') &&
            event.timestamp > currentTimeNs - 1000000000 &&
            event.timestamp <= currentTimeNs
        );

        // Update cursor position and state
        if (currentEvent) {
            mouseCursor.style.display = 'block';

            const displayWidth = mouseDisplay.clientWidth;
            const displayHeight = mouseDisplay.clientHeight;
            // TODO: Update these values based on actual screen resolution
            const screenWidth = 1920;
            const screenHeight = 1080;

            const x = ((currentEvent.x || 0) / screenWidth) * displayWidth;
            const y = ((currentEvent.y || 0) / screenHeight) * displayHeight;

            mouseCursor.style.left = `${x}px`;
            mouseCursor.style.top = `${y}px`;

            // Update cursor appearance based on state and event type
            if (stateEvent && stateEvent.buttons && stateEvent.buttons.length > 0) {
                // Mouse button is pressed
                mouseCursor.style.backgroundColor = 'blue';
                mouseCursor.style.width = '12px';
                mouseCursor.style.height = '12px';
            } else if (lastEvent && lastEvent.event_type === 'move') {
                // Mouse is moving
                mouseCursor.style.backgroundColor = 'green';
                mouseCursor.style.width = '10px';
                mouseCursor.style.height = '10px';
            } else {
                // Default state
                mouseCursor.style.backgroundColor = 'red';
                mouseCursor.style.width = '10px';
                mouseCursor.style.height = '10px';
            }
        } else {
            mouseCursor.style.display = 'none';
        }

        // Clear the mouse display (keeping the cursor)
        function ensureMouseCursorExists() {
            let cursor = mouseDisplay.querySelector('#mouse-cursor');
            if (!cursor) {
                cursor = document.createElement('div');
                cursor.id = 'mouse-cursor';
                // set any default styles here
                mouseDisplay.appendChild(cursor);
            }
            return cursor;
        }
        const cursor = ensureMouseCursorExists();
        mouseDisplay.innerHTML = '';
        mouseDisplay.appendChild(cursor);
        mouseCursor = cursor;

        // Add recent special events if any exist
        if (recentSpecialEvents.length > 0) {
            const eventsDiv = document.createElement('div');
            eventsDiv.className = 'mouse-events';
            eventsDiv.innerHTML = '<h4>Recent Events</h4>';

            recentSpecialEvents.forEach(event => {
                const eventElem = document.createElement('div');

                // Create class based on event type, button, and pressed state
                let classes = ['mouse-event'];
                if (event.event_type === 'click' && event.button) {
                    classes.push(event.button); // 'left', 'right', or 'middle'
                    classes.push(event.pressed ? 'press' : 'release');
                } else if (event.event_type === 'scroll') {
                    classes.push('scroll');
                }
                eventElem.className = classes.join(' ');

                // Create event text
                let eventText = '';
                if (event.event_type === 'click') {
                    const buttonText = event.button ? event.button.charAt(0).toUpperCase() + event.button.slice(1) : '';
                    const actionText = event.pressed ? 'Press' : 'Release';
                    eventText = `${buttonText} ${actionText} at (${event.x}, ${event.y})`;
                } else if (event.event_type === 'scroll') {
                    eventText = `Scroll at (${event.x}, ${event.y})`;
                    if (event.dx != null || event.dy != null) {
                        eventText += ` delta: ${event.dx || 0}, ${event.dy || 0}`;
                    }
                }

                eventElem.textContent = eventText;
                eventsDiv.appendChild(eventElem);
            });

            mouseDisplay.appendChild(eventsDiv);
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
    updateUploadedFilesList();

    // Handle file uploads
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            uploadStatus.textContent = 'Uploading files...';
            uploadStatus.className = 'uploading';

            const formData = new FormData(uploadForm);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                const file_pair = result;

                if (response.ok) {
                    uploadStatus.textContent = 'Files uploaded successfully!';
                    uploadStatus.className = 'success';

                    console.log("Upload result:", file_pair);

                    // Add the new file to our uploaded files list
                    uploadedFiles.push(file_pair);

                    // Update the UI
                    updateUploadedFilesList();

                    // Automatically load the newly uploaded file
                    loadFilePair(file_pair);

                    console.log("Auto-loading newly uploaded file:", file_pair);

                    // Clear the form
                    uploadForm.reset();

                    // Refresh the file list after a successful upload
                    fetchFilePairs();
                } else {
                    uploadStatus.textContent = `Error: ${result.detail || 'Unknown error'}`;
                    uploadStatus.className = 'error';
                }
            } catch (error) {
                uploadStatus.textContent = `Upload failed: ${error.message}`;
                uploadStatus.className = 'error';
            }
        });
    }
});