#!/bin/zsh
#
# Copyright (c) 2022 Dafne-Imaging Team
#

# note: checking signature/notarization: spctl -a -vvv

git pull
git checkout master

APPNAME=Dafne
VERSION=$(python update_version.py | tail -n 1)
ARCH=$(uname -a | sed -E -n 's/.*(arm64|x86_64)$/\1/p')
DMG_NAME=dafne_mac_${VERSION}_$ARCH.dmg
CODESIGN_IDENTITY="Francesco Santini"
USE_ALTOOL=False


NOTARYTOOL() {
  if [ -f /Library/Developer/CommandLineTools/usr/bin/notarytool ]
  then
    /Library/Developer/CommandLineTools/usr/bin/notarytool "$@"
  else 
    xcrun notarytool "$@"
  fi
}

echo $VERSION
pyinstaller dafne_mac.spec --noconfirm
cd dist
echo "Fixing app bundle"
python ../fix_app_bundle_for_mac.py $APPNAME.app
echo "Signing code"
# Sign code outside MacOS
find $APPNAME.app/Contents/Resources -name '*.dylib' | xargs codesign --force -v -s "$CODESIGN_IDENTITY"
find $APPNAME.app/Contents/Resources -name '*.so' | xargs codesign --force -v -s "$CODESIGN_IDENTITY"
# sign the app
codesign --deep --force -v -s "$CODESIGN_IDENTITY" $APPNAME.app
find $APPNAME.app/Contents -path '*bin/*' | xargs codesign --force -o runtime --timestamp --entitlements ../entitlements.plist -v -s "$CODESIGN_IDENTITY"

# Resign the app with the correct entitlement
codesign --force -o runtime --timestamp --entitlements ../entitlements.plist -v -s "$CODESIGN_IDENTITY" $APPNAME.app

echo "Creating DMG"
# Create-dmg at some point stopped working because of a newline prepended to the mount point. If this fails, check for this bug.
create-dmg --volname "Dafne" --volicon $APPNAME.app/Contents/Resources/dafne_icon.icns \
	 --eula $APPNAME.app/Contents/Resources/LICENSE --background ../../icons/mac_installer_bg.png \
	 --window-size 420 220 --icon-size 64 --icon $APPNAME.app 46 31 \
	 --app-drop-link 236 90 "$DMG_NAME" $APPNAME.app
codesign -s "$CODESIGN_IDENTITY" "$DMG_NAME"
echo "Notarizing app"

if [ "$USE_ALTOOL" = "True" ]
then
  # make sure that the credentials are stored in keychain with
  # xcrun altool --store-password-in-keychain-item "AC_PASSWORD" -u "<username>" -p "<password>"
  # Note: password is a "app-specific password" created in the appleID site to bypass 2FA
  xcrun altool --notarize-app \
    --primary-bundle-id "network.dafne.dafne" --password "@keychain:AC_PASSWORD" \
    --file "$DMG_NAME"

  echo "Check the status in 1 hour or so with:"
  echo 'xcrun altool --notarization-history 0 -p "@keychain:AC_PASSWORD"'
  echo 'If there are any errors check'
  echo 'xcrun altool --notarization-info <UUID> -p "@keychain:AC_PASSWORD"'
  echo 'if successful, staple the ticket running'
  echo "xcrun stapler staple $DMG_NAME"
else
  # alternative with notarytool
  # store credentials:
  # xcrun notarytool store-credentials "AC_PASSWORD" --apple-id <apple_id> --password <password> --team-id <team_id>
  echo "This can take up to 1 hour"
  NOTARYTOOL submit "$DMG_NAME" --wait --keychain-profile "AC_PASSWORD"
  # This will wait for the notarization to complete
  echo 'If failed, save log file with:'
  echo 'xcrun notarytool log <request_uuid> --keychain-profile "AC_PASSWORD" notarization.log'
  xcrun stapler staple "$DMG_NAME"
fi
