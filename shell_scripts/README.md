# Shell Scripts

These files are shell scripts that process pcap files from IoT dataset. Pcap files are assumed to be in pcap_files folder. Each pcap file is processed by the Cisco [Joy tool](https://github.com/cisco/joy). The processing ivolved capturing flows from pcap files based on MAC addresses.

1. For each pcap file:
  - For each MAC address:
    - Generate a JSON file that has all flows for that specific MAC (IoT device) for that specific pcap file.

Produced files are saved in json_files folder. For more information, check Algorithm 2 in the paper.
