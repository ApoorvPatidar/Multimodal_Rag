// Front-end interaction logic for chat & PDF upload

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sendBtn = document.getElementById('sendBtn');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');
const pdfInput = document.getElementById('pdfInput');
const pdfSelectBtn = document.getElementById('pdfSelectBtn');
const newChatBtn = document.getElementById('newChatBtn');
const imageInput = document.getElementById('imageInput');
const imageSelectBtn = document.getElementById('imageSelectBtn');
const resetBtn = document.getElementById('resetActive');
const attachmentsBar = document.getElementById('attachmentsBar');
const sidebarToggleTop = document.getElementById('sidebarToggleTop');
const chatHistory = document.getElementById('chatHistory');

let uploading = false;

// Toast notification helper
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 24px;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  `;
  
  if (type === 'success') {
    toast.style.backgroundColor = '#10b981';
  } else if (type === 'error') {
    toast.style.backgroundColor = '#ef4444';
  } else {
    toast.style.backgroundColor = '#3b82f6';
  }
  
  document.body.appendChild(toast);
  
  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Add CSS animation
if (!document.getElementById('toast-styles')) {
  const style = document.createElement('style');
  style.id = 'toast-styles';
  style.textContent = `
    @keyframes slideIn {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(100%); opacity: 0; }
    }
  `;
  document.head.appendChild(style);
}


function scrollToBottom(){
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(content, role){
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role} fade-in`;
  wrapper.innerHTML = `<div class="avatar">${role === 'user' ? 'U':'AI'}</div><div class="message-content">${content}</div>`;
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function setSendEnabled(){
  const msg = messageInput.value.trim();
  sendBtn.disabled = !msg || uploading;
}

sidebarToggle?.addEventListener('click', () => { sidebar.classList.toggle('collapsed'); });
sidebarToggleTop?.addEventListener('click', () => { sidebar.classList.toggle('collapsed'); });

messageInput?.addEventListener('input', () => {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
  setSendEnabled();
});

messageInput?.addEventListener('keydown', (e)=>{
  if (e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn?.addEventListener('click', sendMessage);

pdfSelectBtn?.addEventListener('click', ()=> pdfInput.click());
pdfInput?.addEventListener('change', handlePdfUpload);

newChatBtn?.addEventListener('click', async ()=>{
  await fetch(APP_CONFIG.newChatUrl, {method:'POST'});
  location.reload();
});

resetBtn?.addEventListener('click', async ()=>{
  await fetch(APP_CONFIG.resetActiveUrl, {method:'POST'});
  location.reload();
});
imageSelectBtn?.addEventListener('click', ()=> imageInput.click());
imageInput?.addEventListener('change', handleImageUpload);

async function handlePdfUpload(e){
  const file = e.target.files[0];
  if(!file) return;
  uploading = true; setSendEnabled();
  showToast('Uploading PDF...', 'info');
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch(APP_CONFIG.uploadPdfUrl, {method:'POST', body:formData});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    addOrUpdatePdfChip(data.pdf_name);
    showToast(data.message || `PDF uploaded: ${data.pdf_name}`, 'success');
  } catch(err){
    console.error(err);
    showToast(`PDF upload failed: ${err.message}`, 'error');
  } finally {
    uploading = false; setSendEnabled();
    pdfInput.value = '';
  }
}

async function handleImageUpload(e){
  const file = e.target.files[0];
  if(!file) return;
  showToast('Uploading image...', 'info');
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch(APP_CONFIG.uploadImageUrl, {method:'POST', body:formData});
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || 'Image upload failed');
    refreshImageChips(data.images);
    showToast(data.message || `Image uploaded: ${file.name}`, 'success');
  } catch(err){
    console.error(err);
    showToast(`Image upload failed: ${err.message}`, 'error');
  } finally {
    imageInput.value='';
  }
}

async function sendMessage(){
  const message = messageInput.value.trim();
  if(!message) return;
  addMessage(message, 'user');
  messageInput.value='';
  messageInput.style.height='auto';
  setSendEnabled();
  const placeholder = 'Thinking...';
  const aiWrapper = document.createElement('div');
  aiWrapper.className = 'message assistant fade-in';
  aiWrapper.innerHTML = `<div class="avatar">AI</div><div class="message-content" id="pendingAnswer">${placeholder}</div>`;
  chatMessages.appendChild(aiWrapper);
  scrollToBottom();
  try {
    const res = await fetch(APP_CONFIG.chatUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Chat error');
    aiWrapper.querySelector('#pendingAnswer').textContent = data.answer;
  } catch(err){
    aiWrapper.querySelector('#pendingAnswer').textContent = `Error: ${err.message}`;
  }
}

// Initial state
setSendEnabled();

// Chat history actions (activate / delete / rename)
chatHistory?.addEventListener('click', async (e)=>{
  const btn = e.target;
  const item = btn.closest('.chat-item');
  if(!item) return;
  const chatId = item.getAttribute('data-chat-id');
  const action = btn.getAttribute('data-action');
  if(!action) return;
  if(action === 'activate'){
    await fetch(APP_CONFIG.setActiveChatUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({chat_id: chatId})});
    location.reload();
  } else if(action === 'delete'){
    if(!confirm('Delete this chat?')) return;
    await fetch(APP_CONFIG.deleteChatUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({chat_id: chatId})});
    location.reload();
  } else if(action === 'rename'){
    const newName = prompt('New chat name:');
    if(!newName) return;
    await fetch(APP_CONFIG.renameChatUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({chat_id: chatId, name: newName})});
    location.reload();
  }
});

function addOrUpdatePdfChip(name){
  if(!attachmentsBar) return;
  let chip = attachmentsBar.querySelector('.attachment-chip.pdf');
  if(!chip){
    chip = document.createElement('span');
    chip.className = 'attachment-chip pdf';
    chip.setAttribute('data-type','pdf');
    attachmentsBar.prepend(chip);
  }
  chip.innerHTML = `📄 ${name} <button class="chip-remove" aria-label="Remove PDF" title="Remove">✕</button>`;
}

function refreshImageChips(images){
  if(!attachmentsBar) return;
  attachmentsBar.querySelectorAll('.attachment-chip.image').forEach(c=>c.remove());
  images.forEach(im => {
    const chip = document.createElement('span');
    chip.className = 'attachment-chip image';
    chip.setAttribute('data-type','image');
    chip.setAttribute('data-name', im.name);
    chip.innerHTML = `🖼️ ${im.name} <button class="chip-remove" aria-label="Remove image" title="Remove">✕</button>`;
    attachmentsBar.appendChild(chip);
  });
}

attachmentsBar?.addEventListener('click', async (e)=>{
  const btn = e.target.closest('.chip-remove');
  if(!btn) return;
  const chip = btn.closest('.attachment-chip');
  if(!chip) return;
  const type = chip.getAttribute('data-type');
  if(type === 'pdf'){
    await fetch(APP_CONFIG.removePdfUrl, {method:'POST'});
    chip.remove();
  } else if(type === 'image'){
    const name = chip.getAttribute('data-name');
    await fetch(APP_CONFIG.removeImageUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
    chip.remove();
  }
});
