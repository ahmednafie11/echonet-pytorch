# Overwrite the config file
echo "Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519" > ~/.ssh/config

# Set permissions
chmod 600 ~/.ssh/config
