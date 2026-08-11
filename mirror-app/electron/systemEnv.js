// Environment for spawning SYSTEM binaries (nmcli, bluetoothctl, unzip) from inside an AppImage.
//
// THE BUG THIS EXISTS TO PREVENT: an AppImage is a self-contained bundle that ships its own copies of
// glibc-adjacent libraries, and its runtime injects paths into the environment so the bundled Electron finds
// them — `LD_LIBRARY_PATH`, `XDG_DATA_DIRS`, `GSETTINGS_SCHEMA_DIR`, `GST_PLUGIN_SYSTEM_PATH`, and friends,
// all pointing INSIDE the mount at $APPDIR. Child processes inherit that environment. So a system binary
// spawned from the app resolves its libraries against the AppImage instead of the OS and dies with errors
// like "version `GLIBC_2.38' not found" or "symbol lookup error" — while the exact same command works
// perfectly in a terminal on the same machine.
//
// For this app that would mean: Wi-Fi scanning, joining a network, the hotspot, Bluetooth tethering and OTA
// unpacking all fail on a real Ubuntu unit, and ONLY on the packaged AppImage — never in `electron:dev`,
// never in tests. That is the worst possible failure shape, so it is handled once, here, rather than being
// rediscovered per call site.
//
// The AppImage runtime helpfully preserves the pre-launch values as `<VAR>_ORIG`, which is what makes a
// faithful restore possible: put the original back if there was one, otherwise remove the variable entirely.

// Variables the AppImage runtime prepends its own paths to. Anything left pointing at $APPDIR will make a
// system binary load the bundle's libraries or data files instead of the OS's.
const INJECTED = [
  'LD_LIBRARY_PATH',
  'LD_PRELOAD',
  'PYTHONPATH',
  'PYTHONHOME',
  'PERLLIB',
  'PERL5LIB',
  'GSETTINGS_SCHEMA_DIR',
  'XDG_DATA_DIRS',
  'XDG_CONFIG_DIRS',
  'GTK_PATH',
  'GTK_EXE_PREFIX',
  'GTK_DATA_PREFIX',
  'GDK_PIXBUF_MODULE_FILE',
  'GDK_PIXBUF_MODULEDIR',
  'GI_TYPELIB_PATH',
  'QT_PLUGIN_PATH',
  'GST_PLUGIN_SYSTEM_PATH',
  'GST_PLUGIN_SYSTEM_PATH_1_0',
  'GST_PLUGIN_PATH',
  'GST_PLUGIN_PATH_1_0',
  'LIBRARY_PATH',
  'ALSA_CONFIG_PATH',
]

/** True when running from an AppImage mount, i.e. when the environment needs cleaning. */
function insideAppImage(env = process.env) {
  return Boolean(env.APPDIR || env.APPIMAGE)
}

/**
 * A copy of the environment safe to hand a system binary.
 *
 * Outside an AppImage this is the environment unchanged — a `.deb` install, `electron:dev` and the test
 * suite all run against the OS's own libraries already, and quietly rewriting their environment would make
 * the packaged and unpackaged builds behave differently for no reason.
 */
function systemEnv(env = process.env) {
  if (!insideAppImage(env)) return { ...env }
  const cleaned = { ...env }
  for (const name of INJECTED) {
    const original = cleaned[`${name}_ORIG`]
    if (typeof original === 'string' && original !== '') cleaned[name] = original
    else delete cleaned[name]
    delete cleaned[`${name}_ORIG`]
  }
  // APPDIR itself is how a child would find the bundle at all; nothing outside the app should see it.
  delete cleaned.APPDIR
  return cleaned
}

module.exports = { INJECTED, insideAppImage, systemEnv }
