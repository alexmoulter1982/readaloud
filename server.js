import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { JSONFilePreset } from 'lowdb/node';
import { extract } from '@extractus/article-extractor';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const DATA_DIR = process.env.DATA_DIR || __dirname;

// --- Startup checks ---
const requiredEnvVars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'];
const missing = requiredEnvVars.filter(key => !process.env[key] || process.env[key].startsWith('your_'));

if (missing.length > 0) {
  console.warn('\n⚠  WARNING: The following API keys are missing or unset in .env:');
  missing.forEach(key => console.warn(`   - ${key}`));
  console.warn('   Some features will not work until these are configured.\n');
}

// --- Ensure /recordings directory exists ---
const recordingsDir = path.join(DATA_DIR, 'recordings');
if (!fs.existsSync(recordingsDir)) {
  fs.mkdirSync(recordingsDir, { recursive: true });
  console.log('Created /recordings directory.');
}

// --- Init DB ---
const defaultData = { articles: [] };
const db = await JSONFilePreset(path.join(DATA_DIR, 'db.json'), defaultData);

// --- Express setup ---
const app = express();

app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: false }));

// --- Simple password protection ---
const APP_PASSWORD = process.env.APP_PASSWORD;
if (APP_PASSWORD) {
  app.get('/login', (req, res) => {
    res.send(`<!DOCTYPE html><html><head><title>ReadAloud - Login</title>
    <style>body{font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#111;color:#fff;}
    form{display:flex;flex-direction:column;gap:12px;width:280px;}
    input{padding:10px;border-radius:6px;border:1px solid #444;background:#222;color:#fff;font-size:15px;}
    button{padding:10px;border-radius:6px;border:none;background:#3b82f6;color:#fff;font-size:15px;cursor:pointer;}
    .err{color:#f87171;font-size:13px;}</style></head>
    <body><form method="POST" action="/login">
    <h2 style="margin:0">ReadAloud</h2>
    ${req.query.error ? '<p class="err">Wrong password.</p>' : ''}
    <input type="password" name="password" placeholder="Password" autofocus/>
    <button type="submit">Enter</button>
    </form></body></html>`);
  });

  app.post('/login', (req, res) => {
    if (req.body.password === APP_PASSWORD) {
      res.setHeader('Set-Cookie', `ra_auth=${APP_PASSWORD}; Path=/; HttpOnly; SameSite=Lax`);
      res.redirect('/');
    } else {
      res.redirect('/login?error=1');
    }
  });

  app.use((req, res, next) => {
    if (req.path === '/login') return next();
    const cookie = req.headers.cookie || '';
    const authed = cookie.split(';').some(c => c.trim() === `ra_auth=${APP_PASSWORD}`);
    if (!authed) return res.redirect('/login');
    next();
  });
}

app.use(express.static(path.join(__dirname, 'public')));
app.use('/recordings', express.static(recordingsDir));

// Remove non-substantive elements (captions, media credits, pull quotes, etc.) before stripping HTML
function removeJunk(html) {
  // Remove entire elements that are typically captions or non-article content
  return html
    // figcaption (photo captions)
    .replace(/<figcaption[\s\S]*?<\/figcaption>/gi, '')
    // elements whose class/id contains caption, credit, cutline, byline, photo-, image-caption, subscribe, newsletter, ad-, promo
    .replace(/<[^>]+(?:class|id)="[^"]*(?:caption|credit|cutline|photo-credit|image-credit|subscribe|newsletter|ad-|promo|pull-quote|related|tag|topic)[^"]*"[^>]*>[\s\S]*?<\/\w+>/gi, '')
    // <cite> inside figures
    .replace(/<figure[\s\S]*?<\/figure>/gi, '')
    // inline styles / script / style blocks
    .replace(/<(script|style|noscript|aside|nav|footer|header)[^>]*>[\s\S]*?<\/\1>/gi, '');
}

// Strip HTML tags from a string
function stripHtml(html) {
  return html.replace(/<[^>]*>/g, ' ').replace(/\s{2,}/g, ' ').trim();
}

// Convert a title into a safe filename slug
function slugify(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

// Escape special characters for XML text nodes and attribute values
function escapeXml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

// Format a date string as RFC 822 (required by RSS 2.0)
function toRfc822(dateStr) {
  return new Date(dateStr).toUTCString();
}

// Split text into chunks ≤ maxLen chars, breaking at sentence boundaries
function splitIntoChunks(text, maxLen = 4000) {
  if (text.length <= maxLen) return [text];

  const chunks = [];
  let pos = 0;

  while (pos < text.length) {
    if (text.length - pos <= maxLen) {
      chunks.push(text.slice(pos));
      break;
    }

    // Scan backwards from the limit for a sentence boundary
    let splitAt = -1;
    for (let i = pos + maxLen; i > pos + maxLen - 600 && i > pos; i--) {
      const ch = text[i];
      const next = text[i + 1];
      if ((ch === '.' || ch === '!' || ch === '?') && (next === ' ' || next === '\n' || next === undefined)) {
        splitAt = i + 1;
        break;
      }
      if (ch === '\n') {
        splitAt = i;
        break;
      }
    }

    // Fall back to word boundary
    if (splitAt === -1) {
      for (let i = pos + maxLen; i > pos + maxLen - 200 && i > pos; i--) {
        if (text[i] === ' ') { splitAt = i; break; }
      }
    }

    // Hard split as last resort
    if (splitAt === -1) splitAt = pos + maxLen;

    chunks.push(text.slice(pos, splitAt).trim());
    pos = splitAt;
    while (pos < text.length && text[pos] === ' ') pos++;
  }

  return chunks.filter(c => c.length > 0);
}

// --- Routes ---
app.get('/api/articles', async (req, res) => {
  await db.read();
  res.json(db.data.articles);
});

app.post('/api/extract', async (req, res) => {
  const { url } = req.body;

  if (!url || typeof url !== 'string' || !url.startsWith('http')) {
    return res.status(400).json({ error: 'A valid "url" field is required.' });
  }

  let article;
  try {
    article = await extract(url);
  } catch (err) {
    console.error(`[extract] Failed to fetch "${url}":`, err.message);
    return res.status(502).json({ error: `Failed to extract article: ${err.message}` });
  }

  if (!article) {
    return res.status(422).json({ error: 'No article content could be extracted from that URL.' });
  }

  const title = article.title?.trim() || '';
  const rawContent = article.content || '';
  const text = rawContent ? stripHtml(removeJunk(rawContent)) : '';

  if (!title && !text) {
    return res.status(422).json({ error: 'Extraction returned empty title and content.' });
  }

  console.log(`[extract] Title: ${title}`);
  console.log(`[extract] Text preview: ${text.slice(0, 200)}`);

  res.json({ title, text, sourceUrl: url });
});

app.post('/api/ocr', async (req, res) => {
  const { imageData, mimeType } = req.body;

  if (!imageData || typeof imageData !== 'string') {
    return res.status(400).json({ error: '"imageData" (base64 string) is required.' });
  }
  if (!mimeType || typeof mimeType !== 'string') {
    return res.status(400).json({ error: '"mimeType" is required (e.g. "image/jpeg").' });
  }

  let response;
  try {
    response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: { type: 'base64', media_type: mimeType, data: imageData },
            },
            {
              type: 'text',
              text: 'Extract only the article text from this image. Return just the article title on the first line, then the full body text. No UI elements, ads, captions, or navigation.',
            },
          ],
        },
      ],
    });
  } catch (err) {
    console.error('[ocr] Anthropic API error:', err.message);
    return res.status(502).json({ error: `OCR failed: ${err.message}` });
  }

  const raw = response.content?.[0]?.text?.trim() || '';
  if (!raw) {
    return res.status(422).json({ error: 'Anthropic returned no text from the image.' });
  }

  const newlineIdx = raw.indexOf('\n');
  const title = (newlineIdx !== -1 ? raw.slice(0, newlineIdx) : raw).trim();
  const text = (newlineIdx !== -1 ? raw.slice(newlineIdx + 1) : '').trim();

  console.log(`[ocr] Title: ${title}`);
  console.log(`[ocr] Text preview: ${text.slice(0, 200)}`);

  res.json({ title, text });
});

app.post('/api/generate-audio', async (req, res) => {
  const { title, text, sourceUrl, speed } = req.body;

  if (!title || typeof title !== 'string') {
    return res.status(400).json({ error: '"title" is required.' });
  }
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: '"text" is required.' });
  }

  const ttsSpeed = typeof speed === 'number' ? Math.min(Math.max(speed, 0.25), 4.0) : 1.0;
  const timestamp = Date.now();
  const slug = slugify(title) || 'article';
  const audioFilename = `${timestamp}-${slug}.mp3`;
  const audioPath = path.join(recordingsDir, audioFilename);

  // Split long text into ≤4000-char chunks; prepend title only to first chunk
  const bodyChunks = splitIntoChunks(text);
  const ttsChunks = bodyChunks.map((chunk, i) =>
    i === 0 ? `${title}\n\n${chunk}` : chunk
  );

  console.log(`[generate-audio] ${ttsChunks.length} chunk(s) for "${title}"`);

  const mp3Buffers = [];
  for (let i = 0; i < ttsChunks.length; i++) {
    try {
      const response = await openai.audio.speech.create({
        model: 'tts-1-hd',
        voice: 'onyx',
        input: ttsChunks[i],
        speed: ttsSpeed,
        response_format: 'mp3',
      });
      mp3Buffers.push(Buffer.from(await response.arrayBuffer()));
      console.log(`[generate-audio] Chunk ${i + 1}/${ttsChunks.length} done`);
    } catch (err) {
      console.error(`[generate-audio] OpenAI TTS error on chunk ${i + 1}:`, err.message);
      return res.status(502).json({ error: `TTS generation failed on chunk ${i + 1}: ${err.message}` });
    }
  }

  try {
    fs.writeFileSync(audioPath, Buffer.concat(mp3Buffers));
  } catch (err) {
    console.error('[generate-audio] Failed to save MP3:', err.message);
    return res.status(500).json({ error: 'Failed to save audio file.' });
  }

  const id = uuidv4();
  const date = new Date().toISOString();
  const baseUrl = process.env.BASE_URL || `http://localhost:${PORT}`;
  const audioUrl = `${baseUrl}/recordings/${audioFilename}`;

  await db.read();
  db.data.articles.push({ id, title, sourceUrl: sourceUrl || '', date, audioFilename });
  await db.write();

  console.log(`[generate-audio] Saved: ${audioFilename} (${mp3Buffers.length} chunk(s), speed: ${ttsSpeed})`);

  res.json({ id, audioUrl, title });
});

app.delete('/api/articles/:id', async (req, res) => {
  const { id } = req.params;

  await db.read();
  const idx = db.data.articles.findIndex(a => a.id === id);
  if (idx === -1) {
    return res.status(404).json({ error: 'Article not found.' });
  }

  const article = db.data.articles[idx];

  // Delete MP3 from disk (non-fatal if missing)
  try {
    const filePath = path.join(recordingsDir, article.audioFilename);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch (err) {
    console.warn(`[delete] Could not delete file "${article.audioFilename}":`, err.message);
  }

  db.data.articles.splice(idx, 1);
  await db.write();

  console.log(`[delete] Removed "${article.title}" (${id})`);
  res.json({ ok: true });
});

app.get('/api/feed-url', (req, res) => {
  const baseUrl = process.env.BASE_URL || `http://localhost:${PORT}`;
  res.json({ feedUrl: `${baseUrl}/podcast/feed.xml` });
});

app.get('/podcast/feed.xml', async (req, res) => {
  await db.read();
  const articles = db.data.articles;
  const baseUrl = process.env.BASE_URL || `http://localhost:${PORT}`;
  const feedUrl = `${baseUrl}/podcast/feed.xml`;
  const now = new Date().toUTCString();

  const items = articles.map(article => {
    const audioUrl = `${baseUrl}/recordings/${article.audioFilename}`;
    const pubDate = toRfc822(article.date);
    const link = article.sourceUrl ? escapeXml(article.sourceUrl) : escapeXml(audioUrl);

    // Get file size for <enclosure> — fall back to 0 if file is missing
    let fileSize = 0;
    try {
      const filePath = path.join(recordingsDir, article.audioFilename);
      fileSize = fs.statSync(filePath).size;
    } catch {
      // file may not exist yet in edge cases
    }

    return `
    <item>
      <title>${escapeXml(article.title)}</title>
      <link>${link}</link>
      <guid isPermaLink="false">${escapeXml(article.id)}</guid>
      <pubDate>${pubDate}</pubDate>
      <description>${escapeXml(article.title)}</description>
      <itunes:title>${escapeXml(article.title)}</itunes:title>
      <itunes:author>ReadAloud</itunes:author>
      <itunes:explicit>false</itunes:explicit>
      <enclosure url="${escapeXml(audioUrl)}" length="${fileSize}" type="audio/mpeg"/>
    </item>`.trimStart();
  });

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
  xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>ReadAloud</title>
    <link>${escapeXml(baseUrl)}</link>
    <description>Articles converted to audio by ReadAloud</description>
    <language>en-us</language>
    <lastBuildDate>${now}</lastBuildDate>
    <atom:link href="${escapeXml(feedUrl)}" rel="self" type="application/rss+xml"/>
    <itunes:author>ReadAloud</itunes:author>
    <itunes:explicit>false</itunes:explicit>
    <itunes:category text="Technology"/>
    ${items.join('\n    ')}
  </channel>
</rss>`;

  res.setHeader('Content-Type', 'application/rss+xml; charset=utf-8');
  res.send(xml);
});

// --- Start ---
app.listen(PORT, () => {
  console.log(`ReadAloud server running at http://localhost:${PORT}`);
});
