
window.dash_clientside = Object.assign({}, window.dash_clientside, {

    fullscreen: {
  
      open: function(allBtnClicks, closeClick) {
  
        const ctx = window.dash_clientside.callback_context;
  
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {
  
          return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
  
        }
  
   
  
        const propId = ctx.triggered[0].prop_id; // e.g. {"type":"fs-btn","target":"graph-id"}.n_clicks
  
        const first = propId.split('.')[0];
  
   
  
        // Close request
  
        if (first === "fs-modal-close") {
  
          return [false, window.dash_clientside.no_update, window.dash_clientside.no_update];
  
        }
  
   
  
        // Parse pattern-matching button id
  
        let btnId = null;
  
        try {
  
          btnId = JSON.parse(first);
  
        } catch (e) {
  
          return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
  
        }
  
        const targetId = btnId && btnId.target;
  
        if (!targetId) {
  
          return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
  
        }
  
   
  
        // Find the Plotly graph DOM node and clone its figure
  
        const container = document.getElementById(targetId);
  
        if (!container) {
  
          return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
  
        }
  
   
  
        const gd = container.querySelector('.js-plotly-plot');
  
        if (!gd || !window.Plotly || !window.Plotly.Plots) {
  
          // Open with minimal info rather than failing silently
  
          return [true, targetId, null];
  
        }
  
   
  
        // Extract full figure (data, layout, frames)
  
        const fig = window.Plotly.Plots.graphJson(gd);
  
        const title =
  
          (fig && fig.layout && (fig.layout.title && (fig.layout.title.text || fig.layout.title))) || targetId;
  
   
  
        return [true, title, fig];
  
      }
  
    }
  
  });
  
   
  
   
  
   