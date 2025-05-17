-- ** Toggle with F8 key
-- -- Configuration
-- local hotkeyToggle = VK_F8          -- F8 key as hotkey
-- local speedOn = 0.00001                 -- Speed when enabled
-- local speedOff = 1.0                -- Normal speed
-- local isSpeedOn = false            -- Current toggle state

-- -- Function to toggle speedhack
-- function toggleSpeedhack()
--     if isSpeedOn then
--         -- pause()
--         speedhack_setSpeed(speedOff)
--     else
--         -- unpause()
--         speedhack_setSpeed(speedOn)
--     end
--     isSpeedOn = not isSpeedOn
-- end

-- -- Register the hotkey
-- createHotkey(toggleSpeedhack, hotkeyToggle)

-- showMessage("Speedhack toggle loaded. Press F8 to toggle.")


-- ** Toggle with TCP socket (cheat engine does not support TCP sockets natively)

-- -- Configuration
-- local speedOn = 0.00001
-- local speedOff = 1.0
-- local isSpeedOn = false

-- -- Function to toggle speedhack
-- function toggleSpeedhack()
--     if isSpeedOn then
--         speedhack_setSpeed(speedOff)
--     else
--         speedhack_setSpeed(speedOn)
--     end
--     isSpeedOn = not isSpeedOn
-- end

-- -- TCP Server to receive toggle command
-- local socket = require("socket")
-- local server = assert(socket.bind("127.0.0.1", 12345)) -- localhost:12345
-- server:settimeout(0)  -- non-blocking

-- local function checkSocket()
--     local client = server:accept()
--     if client then
--         client:settimeout(1)
--         local line, err = client:receive()
--         if line == "TOGGLE" then
--             toggleSpeedhack()
--         end
--         client:close()
--     end
-- end

-- -- Timer to check socket periodically
-- local socketTimer = createTimer(nil)
-- timer_setInterval(socketTimer, 100)  -- Check every 100ms
-- timer_onTimer(socketTimer, checkSocket)
-- timer_setEnabled(socketTimer, true)

-- showMessage("Socket-based speedhack toggle loaded (port 12345).")


local processName = "SuperHexagon.exe"

if openProcess(processName) then
    print("Successfully attached to " .. processName)
else
    print("Failed to attach to " .. processName)
end


-- ** Toggle with File
local speedOn = 0.00001               -- Speed when enabled
local speedOff = 1.0                  -- Normal speed
local tempdir = os.getenv("TEMP") or os.getenv("TMP")
local signalFile = tempdir .. "\\speedhack_trigger.txt"
local speedActive = false

function checkSignal()
    local f = io.open(signalFile, "r")
    if f then
        local content = f:read("*all")
        f:close()
        
        if content:find("on") and not speedActive then
            speedhack_setSpeed(speedOn)
            print("Speedhack enabled by file content")
            speedActive = true
        elseif content:find("off") and speedActive then
            speedhack_setSpeed(speedOff)
            print("Speedhack disabled by file content")
            speedActive = false
        end
    end
end

local t = createTimer(nil)
timer_setInterval(t, 10)  -- Check every 10 ms
timer_onTimer(t, checkSignal)
timer_setEnabled(t, true)
showMessage("Speedhack content-based toggle enabled. Use file: "..signalFile)