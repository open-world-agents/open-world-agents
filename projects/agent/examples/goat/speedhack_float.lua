local processName = "SuperHexagon.exe"

if openProcess(processName) then
    print("Successfully attached to " .. processName)
else
    print("Failed to attach to " .. processName)
end

-- ** Toggle with File - Float Value
local defaultSpeed = 1.0              -- Default/normal speed
local tempdir = os.getenv("TEMP") or os.getenv("TMP")
local signalFile = tempdir .. "\\speedhack_value.txt"
local currentSpeed = defaultSpeed

function checkSignal()
    local f = io.open(signalFile, "r")
    if f then
        local content = f:read("*all")
        f:close()
        
        -- Trim whitespace and try to convert to number
        content = content:match("^%s*(.-)%s*$")
        local newSpeed = tonumber(content)
        
        if newSpeed and newSpeed > 0 then
            -- Only update if the speed value has changed
            if newSpeed ~= currentSpeed then
                speedhack_setSpeed(newSpeed)
                print("Speedhack set to: " .. newSpeed)
                currentSpeed = newSpeed
            end
        else
            -- Invalid content, revert to default speed if not already
            if currentSpeed ~= defaultSpeed then
                speedhack_setSpeed(defaultSpeed)
                print("Invalid speed value, reverted to default: " .. defaultSpeed)
                currentSpeed = defaultSpeed
            end
        end
    else
        -- File doesn't exist, use default speed if not already
        if currentSpeed ~= defaultSpeed then
            speedhack_setSpeed(defaultSpeed)
            print("Speed file not found, reverted to default: " .. defaultSpeed)
            currentSpeed = defaultSpeed
        end
    end
end

local t = createTimer(nil)
timer_setInterval(t, 20)  -- Check every 20 ms 
timer_onTimer(t, checkSignal)
timer_setEnabled(t, true)
showMessage("Speedhack float value control enabled. Use file: "..signalFile.."\nWrite a float value (e.g., 0.5, 2.0, 10.0)")