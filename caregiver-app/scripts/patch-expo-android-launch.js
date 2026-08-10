const fs = require('fs');
const path = require('path');

const files = [
  path.join(
    __dirname,
    '..',
    'node_modules',
    'expo',
    'node_modules',
    '@expo',
    'cli',
    'build',
    'src',
    'start',
    'platforms',
    'android',
    'AndroidDeviceManager.js',
  ),
  path.join(
    __dirname,
    '..',
    'node_modules',
    'expo',
    'node_modules',
    '@expo',
    'cli',
    'build',
    'src',
    'start',
    'platforms',
    'android',
    'AndroidDeviceManager.js.map',
  ),
];

const from = '.experience.HomeActivity';
const to = '.LauncherActivity';

for (const file of files) {
  if (!fs.existsSync(file)) {
    continue;
  }

  const source = fs.readFileSync(file, 'utf8');
  if (!source.includes(from)) {
    continue;
  }

  fs.writeFileSync(file, source.replaceAll(from, to));
  console.log(`Patched Expo Android launcher in ${path.relative(process.cwd(), file)}`);
}
