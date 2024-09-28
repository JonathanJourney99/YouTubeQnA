import json
import re
import xml.etree.ElementTree as ET
import requests

class YouTube:
    def get_video_id(self, url):
        pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        match = re.findall(pattern, url)
        print(f"Match {match}")
        if match:
            return match[0]
        else:
            raise ValueError("Invalid YouTube URL")

    def get_transcript(self, video_id):
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        response.raise_for_status()

        soup = response.text
        script_tags = re.findall(r"<script.*?>(.*?)</script>", soup, re.DOTALL)

        for script_tag in script_tags:
            if '"captionTracks":' in script_tag:
                match = re.search(r'"captionTracks":(\[.*?\])', script_tag)
                if match:
                    caption_tracks_json = match.group(1)
                    caption_tracks = json.loads(caption_tracks_json)

                    if caption_tracks:
                        transcript_url = caption_tracks[0]["baseUrl"]
                        transcript_response = requests.get(transcript_url)
                        transcript_response.raise_for_status()
                        return transcript_response.text

        raise ValueError("Transcript not found")

    def parse_transcript_xml(self, xml_input):
        # texts = []
        transcript = ""
        try:
            root = ET.fromstring(xml_input)
            for text in root.findall("text"):
                # start = text.get("start")
                # dur = text.get("dur")
                value = text.text.strip() if text.text else ""
                transcript += value + " "
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return None

        return transcript

def main():
    ytService = YouTube()

    video_id = ytService.get_video_id(
        "https://www.youtube.com/watch?v=IPvYjXCsTg8"
        )
    print(f"Video ID: {video_id}")
    raw_transcript_in_xml = ytService.get_transcript(video_id)
        # print(f"Transcript: \n\n{raw_transcript_in_xml}")
    parsed_transcript = ytService.parse_transcript_xml(raw_transcript_in_xml)
    print(f"Parsed Transcript: \n\n{parsed_transcript}")
    print(len(parsed_transcript))

main()