import { useEffect, useRef } from 'react'

// Polling that always calls the latest callback without resetting the interval timer
// every render -- the standard fix for setInterval capturing a stale closure.
export function useInterval(callback, delayMs) {
  const savedCallback = useRef(callback)
  savedCallback.current = callback

  useEffect(() => {
    if (delayMs == null) return
    const id = setInterval(() => savedCallback.current(), delayMs)
    return () => clearInterval(id)
  }, [delayMs])
}
