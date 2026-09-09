const fs=require('fs'); const path=require('path');
const md=require(path.join(process.env.HOME,'.claude-work/skills/mit-slides/node_modules/markdown-it'))({html:true,linkify:true,typographer:false});
const [,, inp, out, title]=process.argv;
const body=md.render(fs.readFileSync(inp,'utf8'));
const css=`body{font-family:system-ui,Helvetica,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;line-height:1.45;color:#222}
h1{color:#A31F34;border-bottom:2px solid #A31F34;padding-bottom:4px} h2{color:#A31F34;margin-top:1.6em} h3{color:#8A8B8C}
table{border-collapse:collapse;margin:0.8em 0;font-size:0.92em} th{background:#A31F34;color:#fff;padding:4px 10px;text-align:left}
td{padding:4px 10px;border-bottom:1px solid #ccc;vertical-align:top} code{background:#F4F2EE;padding:0 4px;border-radius:3px}
strong{color:#A31F34} .wrap{overflow-x:auto}`;
fs.writeFileSync(out,`<!doctype html><html><head><meta charset="utf-8"><title>${title}</title><style>${css}</style></head><body><div class="wrap">${body}</div></body></html>`);
console.log('wrote',out);
