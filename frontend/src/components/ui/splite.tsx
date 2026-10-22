'use client'

import { useEffect, useRef, useState } from 'react'
import { Brain } from 'lucide-react'

interface SplineSceneProps {
    scene: string
    className?: string
    onLoad?: () => void
}

export function SplineScene({ scene, className, onLoad }: SplineSceneProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [hasError, setHasError] = useState(false)

    useEffect(() => {
        let isMounted = true
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let splineApp: any = null

        async function initSpline() {
            if (!canvasRef.current) return
            try {
                setIsLoading(true)
                setHasError(false)

                // Dynamically import @splinetool/runtime directly on client
                const { Application } = await import('@splinetool/runtime')

                if (!isMounted || !canvasRef.current) return

                splineApp = new Application(canvasRef.current)
                await splineApp.load(scene)

                if (isMounted) {
                    setIsLoading(false)
                    onLoad?.()
                }
            } catch (err) {
                console.warn('Failed to load Spline scene:', err)
                if (isMounted) {
                    setIsLoading(false)
                    setHasError(true)
                    // Still notify parent so chat bubble and UI display properly
                    onLoad?.()
                }
            }
        }

        initSpline()

        return () => {
            isMounted = false
            try {
                if (splineApp) {
                    splineApp.dispose()
                }
            } catch {
                // Ignore cleanup errors
            }
        }
    }, [scene, onLoad])

    return (
        <div className={`relative w-full h-full overflow-hidden ${className || ''}`}>
            {isLoading && !hasError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm z-10">
                    <div className="flex flex-col items-center gap-3 text-muted-foreground animate-pulse">
                        <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                        <span className="text-xs">Loading 3D Assistant...</span>
                    </div>
                </div>
            )}

            {hasError ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-center p-6">
                    <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 mb-3 animate-pulse">
                        <Brain className="size-16 text-primary" />
                    </div>
                    <p className="text-sm font-medium text-white/80">XRayBo AI Assistant</p>
                    <p className="text-xs text-white/50 max-w-xs mt-1">3D Assistant Active</p>
                </div>
            ) : (
                <canvas
                    ref={canvasRef}
                    className="w-full h-full block"
                    style={{ opacity: isLoading ? 0 : 1, transition: 'opacity 0.4s ease' }}
                />
            )}
        </div>
    )
}
