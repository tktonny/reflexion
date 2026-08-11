const assert = require('node:assert/strict')
const { test } = require('node:test')

const { insideAppImage, systemEnv } = require('./systemEnv')

test('outside an AppImage the environment is passed through unchanged', () => {
  // A .deb install, electron:dev and CI already run against the OS's own libraries; rewriting their
  // environment would make packaged and unpackaged builds behave differently for no reason.
  const env = { PATH: '/usr/bin', LD_LIBRARY_PATH: '/opt/mine/lib', HOME: '/home/mirror' }
  assert.deepEqual(systemEnv(env), env)
  assert.equal(insideAppImage(env), false)
})

test('inside an AppImage the injected library path is restored to its original', () => {
  // This is the whole point: nmcli must load the OS's libraries, not the bundle's.
  const cleaned = systemEnv({
    APPDIR: '/tmp/.mount_abc',
    APPIMAGE: '/opt/Reflexion.AppImage',
    LD_LIBRARY_PATH: '/tmp/.mount_abc/usr/lib:/usr/lib',
    LD_LIBRARY_PATH_ORIG: '/usr/lib',
    PATH: '/usr/bin',
  })
  assert.equal(cleaned.LD_LIBRARY_PATH, '/usr/lib')
  assert.equal('LD_LIBRARY_PATH_ORIG' in cleaned, false)
  assert.equal('APPDIR' in cleaned, false)
  assert.equal(cleaned.PATH, '/usr/bin')
})

test('an injected variable with no original is removed rather than left pointing into the bundle', () => {
  // The common case: the variable did not exist before launch, so "restore" means "delete".
  const cleaned = systemEnv({
    APPDIR: '/tmp/.mount_abc',
    LD_LIBRARY_PATH: '/tmp/.mount_abc/usr/lib',
    GSETTINGS_SCHEMA_DIR: '/tmp/.mount_abc/usr/share/glib-2.0/schemas',
    XDG_DATA_DIRS: '/tmp/.mount_abc/usr/share:/usr/share',
  })
  for (const name of ['LD_LIBRARY_PATH', 'GSETTINGS_SCHEMA_DIR', 'XDG_DATA_DIRS']) {
    assert.equal(name in cleaned, false, `${name} should be removed`)
  }
})

test('an empty original counts as "no original" and is removed', () => {
  const cleaned = systemEnv({ APPDIR: '/tmp/.mount_abc', LD_LIBRARY_PATH: '/tmp/x/lib', LD_LIBRARY_PATH_ORIG: '' })
  assert.equal('LD_LIBRARY_PATH' in cleaned, false)
})

test('no bundle path survives anywhere in the cleaned environment', () => {
  // The regression that would break Wi-Fi on a real unit is ANY leftover $APPDIR reference, so assert on the
  // whole result rather than on the variables we happened to think of.
  const mount = '/tmp/.mount_reflexA1'
  const cleaned = systemEnv({
    APPDIR: mount,
    APPIMAGE: '/opt/Reflexion.AppImage',
    LD_LIBRARY_PATH: `${mount}/usr/lib`,
    PERLLIB: `${mount}/usr/share/perl5`,
    GI_TYPELIB_PATH: `${mount}/usr/lib/girepository-1.0`,
    QT_PLUGIN_PATH: `${mount}/usr/plugins`,
    GST_PLUGIN_SYSTEM_PATH_1_0: `${mount}/usr/lib/gstreamer-1.0`,
    GDK_PIXBUF_MODULE_FILE: `${mount}/usr/lib/gdk-pixbuf/loaders.cache`,
    PATH: '/usr/bin:/bin',
    HOME: '/home/mirror',
  })
  for (const [name, value] of Object.entries(cleaned)) {
    assert.equal(String(value).includes(mount), false, `${name} still points into the AppImage mount`)
  }
  // APPIMAGE is the one exception: shellUpdates needs it to know which file to replace, and it is a path to
  // the executable rather than into the mounted bundle.
  assert.equal(cleaned.APPIMAGE, '/opt/Reflexion.AppImage')
  assert.equal(cleaned.PATH, '/usr/bin:/bin')
  assert.equal(cleaned.HOME, '/home/mirror')
})

test('a unit that only sets APPIMAGE is still treated as an AppImage', () => {
  assert.equal(insideAppImage({ APPIMAGE: '/opt/x.AppImage' }), true)
  assert.equal(insideAppImage({ APPDIR: '/tmp/.mount_x' }), true)
})
