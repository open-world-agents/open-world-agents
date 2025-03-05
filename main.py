from pathlib import Path

from mcap_owa.writer import Writer as OWAWriter

from owa_env_desktop.keyboard_mouse.msg import KeyboardEvent


def main():
    output_file = Path("output.mcap")
    stream = output_file.open("wb")
    with OWAWriter(stream) as writer:
        topic = "/chatter"
        event = KeyboardEvent(event_type="press", vk=1)
        for i in range(0, 10):
            publish_time = i
            writer.write_message(topic, event, publish_time=publish_time)


if __name__ == "__main__":
    main()
