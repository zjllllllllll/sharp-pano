// A single pending load per key; no byte/mesh duplication on rapid switches.
export class ViewCache {
  constructor(dispose) {this.entries=new Map();this.dispose=dispose;this.generation=0;}
  peek(key) {return this.entries.get(key)?.value;}
  get(key,loader) {
    const existing=this.entries.get(key);if(existing) return existing.promise;
    const generation=this.generation, entry={value:null};
    entry.promise=Promise.resolve().then(loader).then(value=>{
      if(generation!==this.generation || this.entries.get(key)!==entry) {this.dispose(value);throw new Error('预加载任务已失效');}
      entry.value=value;return value;
    }).catch(error=>{if(this.entries.get(key)===entry)this.entries.delete(key);throw error;});
    this.entries.set(key,entry);return entry.promise;
  }
  clear() {this.generation++;for(const entry of this.entries.values())if(entry.value)this.dispose(entry.value);this.entries.clear();}
  delete(key) {const entry=this.entries.get(key);this.entries.delete(key);if(entry?.value)this.dispose(entry.value);}
  get ready() {return [...this.entries.values()].filter(e=>e.value).length;}
}
