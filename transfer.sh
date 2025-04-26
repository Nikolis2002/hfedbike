#!/bin/bash

# Prompt the user for input
echo "Select the target server:"
echo "1. green-umpire "
echo "2. xenial-yarn"
read -p "Enter your choice (1 or 2): " choice

# Get the file to transfer
read -p "Enter the path to the file or directory you want to transfer: " filepath

# Validate the file
if [ ! -e "$filepath" ]; then
  echo "Error: File or directory does not exist."
  exit 1
fi

# Set server details based on choice
case $choice in
  1)
    target_host="192.168.1.11"
    target_user="mendel"
    target_path="~/test"
    ;;
  2)
    target_host="192.168.1.13"
    target_user="mendel"
    target_path="~/test"
    ;;
    2)
    target_host="192.168.1.15"
    target_user="nick"
    target_path="~/test"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# Execute the scp command
echo "Transferring file to $target_user@$target_host:$target_path"
scp -r "$filepath" "$target_user@$target_host:$target_path"

# Check the exit status of scp
if [ $? -eq 0 ]; then
  echo "File transfer successful!"
else
  echo "File transfer failed. Please check your input or network."
fi
