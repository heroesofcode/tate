/// <reference lib="webworker" />
import { clientsClaim } from 'workbox-core';
import {
  cleanupOutdatedCaches,
  createHandlerBoundToURL,
  precacheAndRoute,
} from 'workbox-precaching';
import { NavigationRoute, registerRoute } from 'workbox-routing';
import { inboxRequestUrl, SHARE_CACHE } from './lib/share-cache';

declare const self: ServiceWorkerGlobalScope;

self.addEventListener('install', (event) => {
  if (!self.registration.active) event.waitUntil(self.skipWaiting());
});

self.addEventListener('message', (event) => {
  const data = event.data as { type?: string } | null;
  if (data?.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
  if (!isSharePost(event.request)) return;
  event.stopImmediatePropagation();
  event.respondWith(handleShare(event.request));
});

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();
clientsClaim();

registerRoute(new NavigationRoute(createHandlerBoundToURL('index.html')));

function isSharePost(request: Request): boolean {
  if (request.method !== 'POST') return false;
  const share = new URL('share', self.registration.scope);
  return new URL(request.url).pathname === share.pathname;
}

async function handleShare(request: Request): Promise<Response> {
  const redirectTo = new URL(self.registration.scope);
  redirectTo.searchParams.set('shared', '1');
  try {
    const form = await request.formData();
    const files: File[] = [];
    form.forEach((value) => {
      if (value instanceof File) files.push(value);
    });
    const cache = await caches.open(SHARE_CACHE);
    const stale = await cache.keys();
    await Promise.all(stale.map((key) => cache.delete(key)));
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      if (!file) continue;
      const name = file.name || `shared-${index}.tate`;
      const url = inboxRequestUrl(self.registration.scope, index, name);
      await cache.put(
        url,
        new Response(file.stream(), {
          headers: { 'Content-Type': file.type || 'application/octet-stream' },
        }),
      );
    }
  } catch {
    /* open the app even if the share body could not be stored */
  }
  return Response.redirect(redirectTo.href, 303);
}
