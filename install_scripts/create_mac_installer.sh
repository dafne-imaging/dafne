#!/bin/zsh
# note: checking signature/notarization: spctl -a -vvv

git pull
cd dl
git checkout master
git pull
cd ..

APPNAME=Dafne
VERSION=`python update_version.py | tail -n 1`
DMG_NAME=dafne_mac_$VERSION.dmg
CODESIGN_IDENTITY="Francesco"

echo $VERSION
pyinstaller dafne_mac.spec --noconfirm
cd dist
echo "Fixing app bundle"
python ../fix_app_bundle_for_mac.py $APPNAME.app
echo "Signing code"
# Sign code outside MacOS
find $APPNAME.app/Contents/Resources -name '*.dylib' | xargs codesign --force -v -s "$CODESIGN_IDENTITY"
# sign the app
codesign --deep --force -v -s "$CODESIGN_IDENTITY" $APPNAME.app

# Resign the app with the correct entitlement
codesign --force -o runtime --entitlements ../entitlements.plist -v -s "$CODESIGN_IDENTITY" $APPNAME.app

echo "Creating DMG"
create-dmg --volname "Dafne" --volicon $APPNAME.app/Contents/Resources/dafne_icon.icns \
	 --eula $APPNAME.app/Contents/Resources/LICENSE --background ../mac_installer_bg.png \
	 --window-size 420 220 --icon-size 64 --icon $APPNAME.app 46 31 \
	 --app-drop-link 236 90 "$DMG_NAME" $APPNAME.app
codesign -s "$CODESIGN_IDENTITY" "$DMG_NAME"
echo "Notarizing app"
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
#rm /tmp/dafne.dmg
#cp calc_transforms/calc_transforms dafne
#zip -r dafne_mac_$VERSION.zip dafne
