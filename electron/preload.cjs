const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('desktop', {
  config: () => ipcRenderer.invoke('config'),
  sources: () => ipcRenderer.invoke('sources'),
  selectSource: id => ipcRenderer.invoke('select-source', id),
  sample: () => ipcRenderer.invoke('sample'),
  captureSettings: () => ipcRenderer.invoke('capture-settings'),
});
