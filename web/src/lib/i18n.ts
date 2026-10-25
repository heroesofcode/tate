export type Locale = 'en' | 'pt';

const en = {
  libraryLabel: 'Library',
  import: 'Import',
  importChapter: 'Import chapter',
  emptyTitle: 'No chapters yet',
  emptyBody: 'Import a .tate chapter from the processor, or a .cbz/.zip of page images.',
  howTo: 'How to make a chapter',
  segments: '{count} panels',
  segmentOne: '1 panel',
  pages: '{count} pages',
  pageOne: '1 page',
  unprocessed: 'Unprocessed',
  continue: 'Continue',
  continueChapter: 'Continue {title}',
  openChapter: 'Open {title}',
  notStarted: 'Not started',
  delete: 'Delete',
  deleteTitle: 'Delete this chapter?',
  deleteBody: '“{title}” will be removed from this device.',
  deleteChapter: 'Delete {title}',
  cancel: 'Cancel',
  replaceTitle: 'Replace chapter?',
  replaceBody:
    'A chapter with the same title, source, and number of panels is already in your library.',
  replace: 'Replace',
  storageUsed: '{used} used',
  storagePersistent: 'Kept on this device',
  storageTemporary: 'May be cleared by the browser',
  keepStorage: 'Keep on device',
  importing: 'Importing {current} of {total}',
  importDone: 'Imported “{title}”',
  importFailed: 'Could not import “{name}”. {message}',
  dropHint: 'Drop to import',
  back: 'Back',
  endTitle: 'End of chapter',
  endBody: 'You’ve reached the end.',
  backToLibrary: 'Back to library',
  installTitle: 'Install tate',
  installIos: 'On iPhone, tap Share, then Add to Home Screen.',
  installAndroid: 'Open the browser menu and tap Install app.',
  installAction: 'Install app',
  notNow: 'Not now',
  updateTitle: 'Update available',
  updateAction: 'Reload',
  viewGrid: 'Grid',
  viewList: 'List',
  lastRead: 'Last read {time}',
  added: 'Added {time}',
  progressLabel: 'Reading progress',
  readerRegion: 'Chapter',
  missingChapter: 'This chapter is no longer on the device.',
  loading: 'Loading library…',
  dismiss: 'Dismiss',
  errTooLarge: 'This file is too large to import.',
  errUnsafePath: 'The archive contains an unsafe path ({detail}).',
  errNotTate: 'This file is not a tate chapter.',
  errVersion: 'This chapter uses format version {detail}, which this reader does not support.',
  errBadManifest: 'The chapter manifest is invalid ({detail}).',
  errMissingFile: 'The package is missing {detail}.',
  errMissingManifest: 'The package has no manifest.json.',
  errNoImages: 'No page images were found in this archive.',
  errFileTooLarge: 'A file inside the archive is too large.',
  errTooMany: 'The archive has too many files.',
  errBadZip: 'This file could not be read as an archive.',
  errGeneric: 'Something went wrong while importing.',
} as const;

export type MessageKey = keyof typeof en;

const pt: Record<MessageKey, string> = {
  libraryLabel: 'Biblioteca',
  import: 'Importar',
  importChapter: 'Importar capítulo',
  emptyTitle: 'Nenhum capítulo ainda',
  emptyBody:
    'Importe um capítulo .tate do processador, ou um .cbz/.zip com as imagens das páginas.',
  howTo: 'Como preparar um capítulo',
  segments: '{count} painéis',
  segmentOne: '1 painel',
  pages: '{count} páginas',
  pageOne: '1 página',
  unprocessed: 'Sem processar',
  continue: 'Continuar',
  continueChapter: 'Continuar {title}',
  openChapter: 'Abrir {title}',
  notStarted: 'Não iniciado',
  delete: 'Apagar',
  deleteTitle: 'Apagar este capítulo?',
  deleteBody: '“{title}” será removido deste aparelho.',
  deleteChapter: 'Apagar {title}',
  cancel: 'Cancelar',
  replaceTitle: 'Substituir capítulo?',
  replaceBody:
    'Já existe um capítulo com o mesmo título, a mesma origem e a mesma quantidade de painéis.',
  replace: 'Substituir',
  storageUsed: '{used} em uso',
  storagePersistent: 'Mantido neste aparelho',
  storageTemporary: 'O navegador pode apagar',
  keepStorage: 'Manter no aparelho',
  importing: 'Importando {current} de {total}',
  importDone: '“{title}” importado',
  importFailed: 'Não foi possível importar “{name}”. {message}',
  dropHint: 'Solte para importar',
  back: 'Voltar',
  endTitle: 'Fim do capítulo',
  endBody: 'Você chegou ao fim.',
  backToLibrary: 'Voltar à biblioteca',
  installTitle: 'Instalar o tate',
  installIos: 'No iPhone, toque em Compartilhar e depois em Adicionar à Tela de Início.',
  installAndroid: 'Abra o menu do navegador e toque em Instalar app.',
  installAction: 'Instalar app',
  notNow: 'Agora não',
  updateTitle: 'Atualização disponível',
  updateAction: 'Recarregar',
  viewGrid: 'Grade',
  viewList: 'Lista',
  lastRead: 'Lido {time}',
  added: 'Adicionado {time}',
  progressLabel: 'Progresso de leitura',
  readerRegion: 'Capítulo',
  missingChapter: 'Este capítulo não está mais neste aparelho.',
  loading: 'Carregando biblioteca…',
  dismiss: 'Fechar',
  errTooLarge: 'Este arquivo é grande demais para importar.',
  errUnsafePath: 'O arquivo contém um caminho inseguro ({detail}).',
  errNotTate: 'Este arquivo não é um capítulo tate.',
  errVersion: 'Este capítulo usa a versão {detail} do formato, que este leitor não suporta.',
  errBadManifest: 'O manifesto do capítulo é inválido ({detail}).',
  errMissingFile: 'Falta este arquivo no pacote: {detail}.',
  errMissingManifest: 'O pacote não tem manifest.json.',
  errNoImages: 'Nenhuma imagem de página foi encontrada neste arquivo.',
  errFileTooLarge: 'Um arquivo dentro do pacote é grande demais.',
  errTooMany: 'O arquivo tem arquivos demais.',
  errBadZip: 'Não foi possível ler este arquivo como um pacote.',
  errGeneric: 'Algo deu errado na importação.',
};

export const messages: Record<Locale, Record<MessageKey, string>> = { en, pt };

export function detectLocale(language: string): Locale {
  return language.toLowerCase().startsWith('pt') ? 'pt' : 'en';
}

export function lookupMessage(
  tables: Record<string, Record<string, string>>,
  locale: string,
  key: string,
): string {
  return tables[locale]?.[key] || tables.en?.[key] || key;
}

function applyVars(text: string, vars?: Record<string, string | number>): string {
  if (!vars) return text;
  let out = text;
  for (const [key, value] of Object.entries(vars)) out = out.replaceAll(`{${key}}`, String(value));
  return out;
}

export function activeLocale(): Locale {
  const language = typeof navigator === 'undefined' ? 'en' : navigator.language;
  return detectLocale(language || 'en');
}

export function intlLocale(): string {
  return activeLocale() === 'pt' ? 'pt-BR' : 'en';
}

export function t(key: MessageKey, vars?: Record<string, string | number>): string {
  return applyVars(lookupMessage(messages, activeLocale(), key), vars);
}

const errorKey: Record<string, MessageKey> = {
  too_large: 'errTooLarge',
  unsafe_path: 'errUnsafePath',
  not_tate: 'errNotTate',
  version: 'errVersion',
  bad_manifest: 'errBadManifest',
  missing_file: 'errMissingFile',
  missing_manifest: 'errMissingManifest',
  no_images: 'errNoImages',
  file_too_large: 'errFileTooLarge',
  too_many_entries: 'errTooMany',
  bad_zip: 'errBadZip',
};

export function packageErrorText(error: unknown): string {
  if (error && typeof error === 'object' && 'code' in error && typeof error.code === 'string') {
    const detail = 'detail' in error && typeof error.detail === 'string' ? error.detail : '';
    const key = errorKey[error.code] ?? 'errGeneric';
    return t(key, { detail });
  }
  return t('errGeneric');
}
