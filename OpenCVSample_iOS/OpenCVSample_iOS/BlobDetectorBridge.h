//
//  BlobDetectorBridge.h
//  OpenCVSample_iOS
//
//  Created by Luu Tran on 05/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#ifndef BlobDetectorBridge_h
#define BlobDetectorBridge_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface BlobDetectorBridge : NSObject
- (void) initialize;
- (UIImage *) processDetect: (UIImage *) image;

@end

NS_ASSUME_NONNULL_END

#endif /* BlobDetectorBridge_h */
