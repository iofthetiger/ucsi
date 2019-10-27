## Installation

### Key Installation
* Generating new ssh key in the server, path ```~/.ssh/ucsi```

* Copy the ```cat ~/.ssh/ucsi.pub``` result to [this setting page](https://github.com/iofthetiger/ucsi/settings/keys)

* Run the following on server
```
eval "$(ssh-agent -s)";ssh-add -k ~/.ssh/ucsi
```
