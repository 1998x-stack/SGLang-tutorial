"use strict";
(function(){
  const cards=[...document.querySelectorAll(".course-card")];
  const search=document.querySelector("#course-search");
  const chips=[...document.querySelectorAll("[data-filter]")];
  let current="all";
  function update(){
    const query=(search?.value||"").trim().toLocaleLowerCase();
    let shown=0;
    for(const card of cards){
      const categoryOK=current==="all"||card.dataset.category===current;
      const textOK=!query||card.textContent.toLocaleLowerCase().includes(query);
      card.hidden=!(categoryOK&&textOK);
      if(!card.hidden)shown++;
    }
    const count=document.querySelector("#result-count");
    if(count) count.textContent="显示 "+shown+" / "+cards.length+" 章";
  }
  if(search)search.addEventListener("input",update);
  for(const chip of chips)chip.addEventListener("click",()=>{
    current=chip.dataset.filter;
    for(const b of chips)b.setAttribute("aria-pressed",String(b===chip));
    update();
  });
  update();
  document.querySelectorAll("[data-stepper]").forEach(root=>{
    const buttons=[...root.querySelectorAll("[data-step]")];
    const panels=[...root.querySelectorAll("[data-panel]")];
    for(const button of buttons)button.addEventListener("click",()=>{
      const key=button.dataset.step;
      for(const b of buttons)b.setAttribute("aria-pressed",String(b===button));
      for(const panel of panels)panel.hidden=panel.dataset.panel!==key;
    });
  });
})();
