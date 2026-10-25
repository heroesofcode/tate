export interface ToastItem {
  id: number;
  message: string;
  tone: 'info' | 'error';
}

class Toasts {
  items = $state<ToastItem[]>([]);
  next = 1;

  push(message: string, tone: 'info' | 'error' = 'info') {
    const id = this.next;
    this.next += 1;
    this.items = [...this.items, { id, message, tone }].slice(-3);
    const timeout = tone === 'error' ? 6400 : 3400;
    setTimeout(() => this.dismiss(id), timeout);
  }

  dismiss(id: number) {
    this.items = this.items.filter((item) => item.id !== id);
  }
}

export const toasts = new Toasts();
