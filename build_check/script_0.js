
/* 主题：在首次绘制前定下来，避免闪白 */
(function(){
  try{
    var t = localStorage.getItem('wapi_theme');
    var KNOWN = ['light','dark','mono-light','mono-dark','custom'];
    if (KNOWN.indexOf(t) < 0) t = 'mono-light';
    // 铁律：只有「自定义」主题才允许自定义底色生效。
    // 预设主题下历史遗留的 wapi_custom_bg 一律忽略并清掉——
    // 之前出现过「主题=极简白、背景却是深色」的矛盾，根源就在这里。
    var cb = localStorage.getItem('wapi_custom_bg');
    if (t !== 'custom' && cb) {
      try { localStorage.removeItem('wapi_custom_bg'); } catch (e) {}
      cb = null;
    }
    document.documentElement.setAttribute('data-theme', t);
    var a = localStorage.getItem('wapi_accent');
    if (['blue','violet','pink','red','green','orange','cyan'].indexOf(a) >= 0)
      document.documentElement.setAttribute('data-accent', a);
    var p = localStorage.getItem('wapi_pattern');
    if (['lines','grid','dots','cross','waves','weave','veil'].indexOf(p) >= 0)
      document.documentElement.setAttribute('data-pattern', p);
    if (cb && t === 'custom') {
      document.documentElement.style.setProperty('--bg', cb);
      // 亮底/暗底决定卡片与文字明暗（与 JS 侧 applyCustomBg 同算法）
      var h = String(cb).replace('#','');
      if (h.length === 3) h = h.charAt(0)+h.charAt(0)+h.charAt(1)+h.charAt(1)+h.charAt(2)+h.charAt(2);
      if (/^[0-9a-fA-F]{6}$/.test(h)) {
        var c = [0,2,4].map(function(i){ return parseInt(h.substr(i,2),16)/255; })
          .map(function(v){ return v<=0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055,2.4); });
        var lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2];
        document.documentElement.setAttribute('data-bgmode', lum > 0.42 ? 'light' : 'dark');
      }
    }
    // 主题切换后没有自定义底色时，输入框/预览色要跟着主题走
    document.addEventListener('DOMContentLoaded', function(){
      if (window.renderBgPick) renderBgPick();
      var f = window.applyTheme;
      if (typeof f === 'function') {
        window.applyTheme = function(t){ var r = f.apply(this, arguments); if (window.renderBgPick) renderBgPick(); return r; };
      }
    });
  }catch(e){ document.documentElement.setAttribute('data-theme','mono-light'); }
})();
